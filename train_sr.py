# -*- coding:utf-8 -*-
import math
import time
import argparse
import os
import urllib
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop, RandomCrop
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from data import DatasetFromFolder
from models.networks import ShadowRemoval, Discrimator
from models.perceptual import perceptual
from utils import save_checkpoint, rgb2yuv, yuvloss

from ssim import SSIM, CLBase

try:
    from apex.parallel import DistributedDataParallel as DDP
    #from apex.acceleration_utils import *
    from apex import amp
    from apex.multi_tensor_apply import multi_tensor_applier

except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    
parser = argparse.ArgumentParser(description="PyTorch ShadowRemoval")
parser.add_argument("--pretrained", default="checkpoints/pretrained/ShadowRemoval/35_shadowsyns.pth", help="path to folder containing the model")
parser.add_argument("--train", default="/media/opt48/data/gnh/ISTD_Dataset/ISTD_Dataset/train/", help="path to real train dataset")
parser.add_argument("--test", default="/media/opt48/data/gnh/ISTD_Dataset/ISTD_Dataset/test/", help="path to test dataset")
parser.add_argument("--batchSize", default = 4, type = int, help="training batch")
parser.add_argument("--save_model_freq", default=5, type=int, help="frequency to save model")
parser.add_argument("--cuda", default=True, type=bool, help="frequency to save model")
parser.add_argument("--parallel", action = 'store_true', help = "parallel training")
parser.add_argument("--is_hyper", default=1, type=int, help="use hypercolumn or not")
parser.add_argument("--is_training", default=1, help="training or testing")
parser.add_argument("--continue_training", default = True, type=bool,  help="search for checkpoint in the subfolder specified by `task` argument")
parser.add_argument("--lr_g", default = 3e-4, type = float, help="generator learning rate")
parser.add_argument("--lr_d", default = 1e-3, type = float, help = "discrimator learning rate")
parser.add_argument("--epoch", default = 100, type = int, help = "training epoch")
parser.add_argument("--acceleration", default = False, type = bool, help = "activating acceleration or mix precision acceleration")
parser.add_argument("--opt_level", default = "O1", help = "setting the apex mode")
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--distributed", default=False, type=bool)

class loss_function(nn.Module):
  def __init__(self, smoothl1 = True, l1 = False, mse = False, instance_ssim = True, bce = False, perceptual_loss = True, yuv = False):
    super(loss_function, self).__init__()
    print("=========> Building criterion")
    self.loss_function = nn.ModuleDict()
    self.weight = dict()
    
    self.loss_function['loss_smooth_l1'] = nn.SmoothL1Loss() if smoothl1 else None
    self.loss_function['loss_l1'] = nn.L1Loss() if l1 else None
    self.loss_function['loss_mse'] = torch.nn.MSELoss() if mse else None
    self.loss_function['instance_ssim'] = SSIM(reduction = 'mean', window_size = 7, asloss = True) if instance_ssim else None
    self.loss_function['loss_perceptual'] = perceptual() if perceptual_loss else None
    self.loss_function['loss_bce'] = torch.nn.BCELoss() if bce else None
    self.loss_function['loss_yuv'] = yuvloss() if yuv else None
    
    self.weight['loss_smooth_l1'] = 1
    self.weight['loss_l1'] = 1
    self.weight['loss_mse'] = 1
    self.weight['instance_ssim'] = 1
    self.weight['loss_perceptual'] = 0.2
    self.weight['loss_bce'] = 1
    self.weight['loss_yuv'] = 1
    
    if opt.cuda:
        if False:
          for key in self.loss_function.keys():
            if self.loss_function[key] is not None:
              self.loss_function[key] = nn.DataParallel(self.loss_function[key], [0, 1, 2, 3]).cuda()
          
        else:
          for key in self.loss_function.keys():
            if self.loss_function[key] is not None:
              self.loss_function[key] = self.loss_function[key].cuda()
          
    else:
        for key in self.loss_function.keys():
            if self.loss_function[key] is not None:
              self.loss_function[key] = self.loss_function[key].cpu()
          
  
  def forward(self, reconstructed, target, original_r, original_t):
      loss = 0
      for key in self.loss_function.keys():
        if self.loss_function[key] is not None:
          if key == "instance_ssim" or key == "loss_bce":
            loss += self.loss_function[key](original_r, original_t).mul(self.weight[key])
          else:
            loss += self.loss_function[key](reconstructed, target).mul(self.weight[key])     
      return loss
      
def main():

    global opt, name, logger, netG, netD, vgg, curriculum_ssim_mask, curriculum_ssim_clean, loss_mse, rgb2yuv, instance_ssim, loss_bce

    opt = parser.parse_args()

    name = "ShadowRemoval"
    
    print(opt)

    # Tag_ResidualBlocks_BatchSize

    logger = SummaryWriter("./runs_sr/" + time.strftime("/%Y-%m-%d-%H/", time.localtime()))

    cuda = opt.cuda
    
    if 'WORLD_SIZE' in os.environ:
      opt.distributed = int(os.environ['WORLD_SIZE']) > 1
          
    if cuda and not torch.cuda.is_available():

        raise Exception("No GPU found, please run without --cuda")
    
    if opt.distributed:
      opt.gpu = opt.local_rank
      torch.cuda.set_device(opt.gpu)
      torch.distributed.init_process_group(backend='nccl', init_method='env://')
      opt.world_size = torch.distributed.get_world_size()
          
    seed = 1334

    torch.manual_seed(seed)

    if cuda:

        torch.cuda.manual_seed(seed)

    cudnn.benchmark = True
    
    print("==========> Loading datasets")
    train_dataset = DatasetFromFolder(
        opt.train, 
        transform=Compose([ToTensor()]),
        training = True,
        experiments = "ShadowRemoval",
        )
    test_dataset = DatasetFromFolder(
        opt.test, 
        transform=Compose([ToTensor()]),
        training = False,
        experiments = "ShadowRemoval"
        )

    train_data_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=False)


    print("==========> Building model")
    netG = ShadowRemoval(channels =64)
    netD = Discrimator(in_channels = 6, channels = 64, depth = 3)
    
    print("=========> Building criterion")
    loss_smooth_l1 = nn.SmoothL1Loss()
    loss_l1 = nn.L1Loss()
    loss_mse = torch.nn.MSELoss()
    loss_bce = torch.nn.BCELoss()
    loss_perceptual = perceptual() 
    
    instance_ssim = SSIM(reduction = 'mean', window_size = 7)
    rgb2yuv = rgb2yuv()
    curriculum_ssim_mask  = CLBase(lossfunc = nn.BCELoss(reduce = False))
    curriculum_ssim_clean = CLBase()
    
    # optionally copy weights from a checkpoint
    if opt.pretrained and opt.continue_training:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            netG.load_state_dict(weights['state_dict'])
            
        else:
            print("=> no model found at '{}'".format(opt.pretrained))


    print("==========> Setting Optimizer")
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=opt.lr_g, betas = (0.9, 0.999))
    #optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.module.parameters() if opt.parallel else netD.parameters()), lr=opt.lr_d, betas = (0.5, 0.999))
    optimizerD = optim.SGD(filter(lambda p: p.requires_grad, netD.parameters()), lr=opt.lr_d)
    
    
    print("==========> Setting GPU")
    if cuda:
        netG = netG.cuda()
        netD = netD.cuda()
        
        instance_ssim = instance_ssim.cuda()
        loss_smooth_l1 = loss_smooth_l1.cuda()        
        loss_mse = loss_mse.cuda()
        loss_l1 = loss_l1.cuda()
        loss_bce = loss_bce.cuda()
        curriculum_ssim_mask = curriculum_ssim_mask.cuda()
        curriculum_ssim_clean = curriculum_ssim_clean.cuda()
        loss_perceptual = loss_perceptual.cuda()
        rgb2yuv = rgb2yuv.cuda()
        
        if opt.acceleration:
          print("FP 16 Trianing")
          amp.register_float_function(torch, 'sigmoid')
          [netD, netG], [optimizerD, optimizerG] = amp.initialize([netD, netG], [optimizerD, optimizerG], opt_level=opt.opt_level)
        
        if opt.parallel:
          print("Parallel Training")
          netG = nn.DataParallel(netG)
          netD = nn.DataParallel(netD)
        elif opt.distributed:
          netG = DDP(netG, delay_allreduce=True)
          netD = DDP(netD, delay_allreduce=True)
            
    else:
        netG = netG.cpu()
        netD = netD.cpu()
        
        instance_ssim = instance_ssim.cpu()
        loss_smooth_l1 = loss_smooth_l1.cpu()        
        loss_mse = loss_mse.cpu()
        loss_l1 = loss_l1.cpu()
        loss_bce = loss_bce.cpu()
        curriculum_ssim = curriculum_ssim.cpu()
        loss_perceptual = loss_perceptual.cpu()
        rgb2yuv = rgb2yuv.cpu()

    
    lr_schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, opt.epoch, eta_min = 1e-7)
    lr_schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimizerD, opt.epoch, eta_min = 1e-7)

    print("==========> Training")
    for epoch in range(opt.epoch + 1):

        train(train_data_loader, netG, netD, optimizerG, optimizerD, epoch, logger = logger)
        #test(test_data_loader, netG)
        
        if epoch % opt.save_model_freq == 0:
          save_checkpoint(netG, epoch, name, opt)
          
        lr_schedulerG.step()
        lr_schedulerD.step()     
    
    logger.close() 

    
def train(train_data_loader, netG, netD, optimizerG, optimizerD, epoch, logger):

    print("epoch =", epoch, "lr =", optimizerG.param_groups[0]["lr"])

    Wasserstein_D = torch.zeros(1)

    penalty_lambda = 100
    
    psnrs_clean = []
    psnrs_mask = []
    
    ssims_clean = []
    ssims_mask= []

    bces_mask = []
    
    atomAge = 1.0 / len(train_data_loader)

    loss_function_mask = loss_function(smoothl1 = False, l1 = False, mse = False, instance_ssim = False, bce = True, perceptual_loss = False, yuv = False)
    loss_function_clean = loss_function(smoothl1 = False, l1 = True, mse = False, instance_ssim = False, bce = False, perceptual_loss = True, yuv = True)

    for iteration, batch in enumerate(train_data_loader, 1):

        netG.train()

        netD.train()

        optimizerD.zero_grad()

        steps = len(train_data_loader) * (epoch-1) + iteration

        data_shadow, data_mask, data_clean = \
           Variable(batch[0]), \
           Variable(batch[1]), \
           Variable(batch[2], requires_grad=False)

        if opt.cuda:

           data_clean = data_clean.cuda()
           data_mask = data_mask.cuda()
           data_shadow = data_shadow.cuda()

        else:
           data_clean = data_clean.cuda()
           data_mask = data_mask.cuda()
           data_shadow = data_shadow.cuda()
        
        ########################
        # (1) Update D network Every Two Iteration#
        ########################
        
        # train with fake
        predict_clean, predict_mask = netG(data_shadow)
        
        # curriculum learning 
        c_predict_clean, c_data_clean = predict_clean, data_clean
        c_predict_mask, c_data_mask = predict_mask, data_mask
        
        if iteration % 2 == 0:
          for p in netD.parameters():   # reset requires_grad to True
            p.requires_grad = True    # they are set to False again in netG update.

          # train with real
          D_real = netD(torch.cat((data_shadow, c_data_clean), dim=1).detach())
          D_fake = netD(torch.cat((data_shadow, c_predict_clean), dim=1).detach())
          
          # train with gradient penalty and curriculum regularization
          gradient_penalty = calc_gradient_penalty(netD, data_shadow.data, data_clean.data, data_mask.data, c_predict_clean.data.detach(), c_predict_mask.data.detach(), penalty_lambda)
          
          D_loss = 0.5*((D_real-1).mean() + D_fake.mean()) + gradient_penalty
          Wasserstein_D = (D_fake.mean() - D_real.mean())  

          netD.zero_grad()
          
          if opt.acceleration:
            with amp.scale_loss(D_loss, optimizerD) as D_loss_scaled:
              D_loss_scaled.clamp(1e-8, 1e8)
              D_loss_scaled.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizerD), 5)
          
          else:    
            D_loss.backward(retain_graph = True)

          optimizerD.step()    

        
        ########################
        # (2) update G network #
        ########################

        optimizerG.zero_grad()

        for p in netD.parameters(): # reset requires_grad to False

          p.requires_grad = False   # they are set to True again in netD update

        netG.zero_grad()
        
        G_loss = (netD(torch.cat((data_shadow, c_predict_clean), dim=1)) - 1).mean()

        loss_clean = loss_function_clean(reconstructed = c_predict_clean, target = c_data_clean, original_r = predict_clean, original_t = data_clean).mean()
        loss_mask = loss_function_mask(reconstructed = c_predict_mask.clamp(0,1), target = c_data_mask.clamp(0,1), original_r = predict_mask.clamp(0,1), original_t = data_mask.clamp(0,1)).mean()
        
        with torch.no_grad():

          psnr_mask = 10 * torch.log10(1.0 / loss_mse(predict_mask, data_mask.detach())).mean()
          psnr_clean = (10 * torch.log10(1.0 / loss_mse(rgb2yuv(predict_clean)[:,0].unsqueeze(1), rgb2yuv(data_clean)[:,0].unsqueeze(1)).detach())).mean()

          psnrs_mask.append(psnr_mask.mean())
          psnrs_clean.append(psnr_clean.mean())
          
          ssim_mask = 1 - instance_ssim(predict_mask, data_mask).mean()
          ssim_clean = 1 - instance_ssim(rgb2yuv(predict_clean), rgb2yuv(data_clean)).mean()
          
          ssims_mask.append(ssim_mask)
          ssims_clean.append(ssim_clean)
          
          bce_mask = loss_bce(predict_mask.clamp(0,1), data_mask).mean()          
          bces_mask.append(bce_mask)
          
        loss = 100*(0.2*loss_clean) + loss_mask + G_loss
        
        if opt.acceleration:
          with amp.scale_loss(loss, optimizerG) as loss_scaled:
            loss_scaled.clamp(1e-8, 1e8)
            loss_scaled.backward()
          torch.nn.utils.clip_grad_norm_(amp.master_params(optimizerG), 5)
          
        else:      
          loss.backward()

        optimizerG.step()

        if (iteration-1) % 10 == 0:
          print("===> Epoch[{}]({}/{}): Loss_clean:{:.6f}: Loss_mask:{:.6f} : Psnr_clean:{:.2f} : Psnr_mask:{:.2f} : SSIM_clean:{:2f} : SSIM_mask:{:2f} : BCE_mask:{:2f} : Wasserstein:{} : Lr:{:6f}".format(epoch, iteration, len(train_data_loader), loss_clean.mean(), loss_mask.mean(), psnr_clean.mean(), psnr_mask.mean(), ssim_clean.mean(), ssim_mask.mean(), bce_mask.mean(), Wasserstein_D.mean(), optimizerG.param_groups[0]['lr']))
          
          logger.add_scalar('loss', loss.item(), steps)

          logger.add_scalar('psnr', psnr_clean.item(), steps)

          logger.add_scalar('ssim', ssim_clean.item(), steps)
          
          logger.add_scalar('bce', bce_mask.item(), steps)
     
        show = []

        if (iteration-1) % 10 == 0:

            for idx, tensor in enumerate(zip(data_clean.data.cpu(), data_mask.data.cpu().expand_as(data_clean), data_shadow.data.cpu(), predict_clean.data.cpu().clamp(0,1), predict_mask.data.cpu().expand_as(predict_clean).clamp(0,1))):

              if idx >1:

                break

              show.extend([tensor[0], tensor[1], tensor[2], tensor[3], tensor[4]])

            show = torch.stack(show,0)

            show = make_grid(show, nrow = 5, padding = 5)

            logger.add_image('Comparison_nEpochs:{}'.format(epoch), show)

    
def test(test_data_loader, netG):
    psnrs_clean = []
    psnrs_mask = []
    
    ssims_clean = []
    ssims_mask= []

    bces_mask = []
    
    for iteration, batch in enumerate(test_data_loader, 1):

        netG.eval()
        steps = iteration

        data_shadow, data_mask, data_clean = \
           Variable(batch[0]), \
           Variable(batch[1]), \
           Variable(batch[2], requires_grad=False)

        if opt.cuda:

           data_clean = data_clean.cuda()
           data_mask = data_mask.cuda()
           data_shadow = data_shadow.cuda()

        else:
           data_clean = data_clean.cpu()
           data_mask = data_mask.cpu()
           data_shadow = data_shadow.cpu()
           
        predict_clean, predict_mask = netG(data_shadow)
          
        with torch.no_grad():

          psnr_mask = (10 * torch.log10(1.0 / loss_mse(predict_mask, data_mask.detach()))).mean()
          psnr_clean = (10 * torch.log10(1.0 / loss_mse(rgb2yuv(predict_clean)[:,0].unsqueeze(1), rgb2yuv(data_clean)[:,0].unsqueeze(1)).detach())).mean()

          psnrs_mask.append(psnr_mask)
          psnrs_clean.append(psnr_clean)
          
          ssim_mask = 1 - instance_ssim(predict_mask, data_mask).mean()
          ssim_clean = 1 - instance_ssim(rgb2yuv(predict_clean)[:,0].unsqueeze(1), rgb2yuv(data_clean)[:,0].unsqueeze(1)).mean()
          
          ssims_mask.append(ssim_mask)
          ssims_clean.append(ssim_clean)
        
          bce_mask = loss_bce(predict_mask.clamp(0,1), data_mask)          
          bces_mask.append(bce_mask)
          
    print("psnr_mask:{} psnr_clean:{} ssim_mask:{} ssim_clean:{} bce_mask:{}".format(sum(psnrs_mask)/len(psnrs_mask), sum(psnrs_clean)/len(psnrs_clean), sum(ssims_mask)/len(ssims_mask), sum(ssims_clean)/len(ssims_clean), sum(bces_mask)/len(bces_mask)))

def calc_gradient_penalty(netD, data_shadow, data_clean, data_mask, predict_clean, predict_mask, penalty_lambda):

    alpha = torch.rand(1,1).cuda()
    alpha_clean = alpha.expand_as(data_clean)
    interpolates_clean = Variable((alpha_clean * data_clean + (1-alpha_clean) * predict_clean), requires_grad = True)
    inputs = torch.cat((data_shadow, interpolates_clean), dim = 1)
    disc_interpolates = netD(inputs)

    gradients = torch.autograd.grad(outputs = disc_interpolates, inputs=inputs,

                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),

                              create_graph = True,

                              retain_graph = True,

                              only_inputs=True,
                             )[0]

                              
    gradient_penalty = ((gradients.norm(2, dim = 1) - 1)).mean() * penalty_lambda

    return gradient_penalty
    
if __name__ == "__main__":
   os.system('clear')
   main()