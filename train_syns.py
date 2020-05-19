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
from models.networks import ShadowMattingNet, Discrimator
from models.perceptual import perceptual
from utils import save_checkpoint, rgb2yuv

from ssim import SSIM, CLBase

parser = argparse.ArgumentParser(description="PyTorch ShadowRemoval")
parser.add_argument("--pretrained", default="checkpoints/", help="path to folder containing the model")
parser.add_argument("--train", default="./datasets/ISTD_Dataset/ISTD_Dataset/train/", help="path to real train dataset")
parser.add_argument("--test", default="./datasets/ISTD_Dataset/ISTD_Dataset/test/", help="path to test dataset")
parser.add_argument("--batchSize", default = 4, type = int, help="training batch")
parser.add_argument("--save_model_freq", default=5, type=int, help="frequency to save model")
parser.add_argument("--cuda", default=True, type=bool, help="frequency to save model")
parser.add_argument("--parallel", action = 'store_true', help = "parallel training")
parser.add_argument("--is_hyper", default=1, type=int, help="use hypercolumn or not")
parser.add_argument("--is_training", default=1, help="training or testing")
parser.add_argument("--continue_training", action="store_true", help="search for checkpoint in the subfolder specified by `task` argument")
parser.add_argument("--lr_g", default = 3e-4, type = float, help="generator learning rate")
parser.add_argument("--lr_d", default = 1e-3, type = float, help = "discrimator learning rate")
parser.add_argument("--epoch", default = 500, type = int, help = "training epoch")

class loss_function(nn.Module):
  def __init__(self, smoothl1 = True, l1 = False, mse = False, instance_ssim = True, perceptual_loss = True):
    super(loss_function, self).__init__()
    print("=========> Building criterion")
    self.loss_function = nn.ModuleDict()
    self.weight = dict()
    
    self.loss_function['loss_smooth_l1'] = nn.SmoothL1Loss() if smoothl1 else None
    self.loss_function['loss_l1'] = nn.L1Loss() if l1 else None
    self.loss_function['loss_mse'] = torch.nn.MSELoss() if mse else None
    self.loss_function['instance_ssim'] = SSIM(reduction = 'mean', window_size = 7, asloss = True) if instance_ssim else None
    self.loss_function['loss_perceptual'] = perceptual() if perceptual_loss else None
    
    self.weight['loss_smooth_l1'] = 1
    self.weight['loss_l1'] = 1
    self.weight['loss_mse'] = 1
    self.weight['instance_ssim'] = 1
    self.weight['loss_perceptual'] = 1
    
    if opt.cuda:
        if opt.parallel:
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
          if key == "instance_ssim":
            loss += self.loss_function[key](original_r, original_t)
          else:
            loss += self.loss_function[key](reconstructed, target)     
      return loss 
      
      
def main():

    global opt, name, logger, netG, netD, vgg, curriculum_ssim, loss_mse, rgb2yuv, instance_ssim 

    opt = parser.parse_args()
    
    name = "ShadowSyns"
    
    print(opt)

    # Tag_ResidualBlocks_BatchSize

    logger = SummaryWriter("/home/lthpc/gnh/ShadowRemoval/runs_ss/" + time.strftime("/%Y-%m-%d-%H/", time.localtime()))

    cuda = opt.cuda

    if cuda and not torch.cuda.is_available():

        raise Exception("No GPU found, please run without --cuda")

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
        )

    test_dataset = DatasetFromFolder(
        opt.test, 
        transform=Compose([ToTensor()]),
        training = False,
        )

    train_data_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=True)


    print("==========> Building model")
    netG = ShadowMattingNet(channels =64, depth = 9)
    netD = Discrimator(in_channels = 6, channels = 64, depth = 5)

    
    print("=========> Building criterion")
    loss_smooth_l1 = nn.SmoothL1Loss()
    loss_l1 = nn.L1Loss()
    loss_mse = torch.nn.MSELoss()
    instance_ssim = SSIM(reduction = 'mean', window_size = 7)
    curriculum_ssim  = CLBase()
    loss_perceptual = perceptual() 
    rgb2yuv = rgb2yuv()
    
    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            netG.load_state_dict(weights['state_dict_g'])

        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("==========> Setting GPU")
    if cuda:
        if opt.parallel:
          netG = nn.DataParallel(netG, [0, 1, 2, 3]).cuda()
          netD = nn.DataParallel(netD, [0, 1, 2, 3]).cuda()
        
          instance_ssim = nn.DataParallel(instance_ssim, [0, 1, 2, 3]).cuda()
          loss_smooth_l1 = nn.DataParallel(loss_smooth_l1, [0, 1, 2, 3]).cuda()        
          loss_mse = nn.DataParallel(loss_mse, [0, 1, 2, 3]).cuda()
          loss_l1 = nn.DataParallel(loss_l1, [0, 1, 2, 3]).cuda()
          curriculum_ssim = nn.DataParallel(curriculum_ssim, [0, 1, 2, 3]).cuda()
          rgb2yuv = nn.DataParallel(rgb2yuv, [0, 1, 2, 3]).cuda()
        else:
          netG = netG.cuda()
          netD = netD.cuda()
        
          instance_ssim = instance_ssim.cuda()
          loss_smooth_l1 = loss_smooth_l1.cuda()        
          loss_mse = loss_mse.cuda()
          loss_l1 = loss_l1.cuda()
          curriculum_ssim = curriculum_ssim.cuda()
          loss_perceptual = loss_perceptual.cuda()
          rgb2yuv = rgb2yuv.cuda()
    else:
        netG = netG.cpu()
        netD = netD.cpu()
        
        instance_ssim = instance_ssim.cpu()
        loss_smooth_l1 = loss_smooth_l1.cpu()        
        loss_mse = loss_mse.cpu()
        loss_l1 = loss_l1.cpu()
        curriculum_ssim = curriculum_ssim.cpu()
        loss_perceptual = loss_perceptual.cpu()
        rgb2yuv = rgb2yuv.cpu()
    
    print("==========> Setting Optimizer")
    
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.module.parameters() if opt.parallel else netG.parameters()), lr=opt.lr_g, betas = (0.5, 0.99))
    #optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.module.parameters() if opt.parallel else netD.parameters()), lr = opt.lr_d, betas = (0.5, 0.999))
    optimizerD = optim.SGD(filter(lambda p: p.requires_grad, netD.module.parameters() if opt.parallel else netD.parameters()), lr = opt.lr_d)

    
    lr_schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, opt.epoch, eta_min = 1e-7)
    lr_schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimizerD, opt.epoch, eta_min = 1e-7)

   
    print("==========> Training")
    for epoch in range(opt.epoch + 1):

        train(train_data_loader, netG, netD, optimizerG, optimizerD, epoch)
        test(test_data_loader, netG)
        
        if epoch % opt.save_model_freq == 0:
          save_checkpoint(netG, epoch, name)

        lr_schedulerG.step()
        lr_schedulerD.step()     


def calc_gradient_penalty(netD, data_clean, data_mask, data_predict, data_shadow, penalty_lambda):

    alpha = torch.rand(1,1).expand_as(data_clean).cuda()
    
    interpolates = Variable((alpha * data_shadow + (1-alpha) * data_predict), requires_grad = True)

    disc_interpolates = netD(torch.cat((data_clean, interpolates), dim = 1))
    '''   
    interpolates = Variable((alpha * data_shadow + (1-alpha) * data_predict), requires_grad = True)

    disc_interpolates = netD(interpolates)
    '''
    gradients = torch.autograd.grad(outputs = disc_interpolates, inputs=interpolates,

                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),

                              create_graph = True,

                              retain_graph = True,

                              only_inputs=True)[0]

                              
    gradient_penalty = ((gradients.norm(2, dim = 1))-1).pow(2).mean() * penalty_lambda

    return gradient_penalty
    
    
def train(training_data_loader, netG, netD, optimizerG, optimizerD, epoch):

    print("epoch =", epoch, "lr =", optimizerG.param_groups[0]["lr"])

    Wasserstein_D = torch.zeros(1)

    penalty_lambda = 50

    psnrs = []
    ssims = []
    
    atomAge = 1.0 / len(training_data_loader)

    loss_function_main = loss_function(smoothl1 = False, l1 = False, mse = True, instance_ssim = True, perceptual_loss = True)

    for iteration, batch in enumerate(training_data_loader, 1):

        netG.train()

        netD.train()

        optimizerD.zero_grad()

        steps = len(training_data_loader) * (epoch-1) + iteration

        data_clean, data_mask, data_shadow = \
           Variable(batch[0], requires_grad=True), \
           Variable(batch[1], requires_grad=True), \
           Variable(batch[2], requires_grad=False)

        if opt.cuda:

           data_clean = data_clean.cuda()
           data_mask = data_mask.cuda()
           data_shadow = data_shadow.cuda()

        else:
           data_clean = data_clean.cpu()
           data_mask = data_mask.cpu()
           data_shadow = data_shadow.cpu()
        ########################
        # (1) Update D network Every Two Iteration#
        ########################
        
        # train with fake
        data_predict = netG(data_clean, data_mask)
        
        # curriculum learning 
        #c_data_predict, c_data_shadow, weight = curriculum_ssim(data_predict, data_shadow.detach(), epoch + atomAge * iteration)
        c_data_predict, c_data_shadow = data_predict, data_shadow
         
        if iteration % 5 == 0:
          for p in netD.parameters():   # reset requires_grad to True
            p.requires_grad = True    # they are set to False again in netG update.

          # train with real
          D_real = netD(torch.cat((data_clean, c_data_shadow), dim=1))
          D_fake = netD(torch.cat((data_clean, c_data_predict.detach()), dim=1))
          
          #D_real = netD(((c_data_shadow)))
          #D_fake = netD(((c_data_predict.detach())))
          
          # train with gradient penalty and curriculum regularization
          gradient_penalty = calc_gradient_penalty(netD, data_clean.data, data_mask.data, c_data_predict.data, c_data_shadow.data, penalty_lambda)
          
          D_loss = (0.5*((D_real-1).pow(2).mean() + D_fake.pow(2).mean()) + gradient_penalty)
          Wasserstein_D = (D_fake.pow(2).mean() - D_real.pow(2).mean())  

          netD.zero_grad()

          D_loss.backward(retain_graph = True)

          optimizerD.step()    

        
        ########################
        # (2) update G network #
        ########################

        optimizerG.zero_grad()

        for p in netD.parameters(): # reset requires_grad to False

          p.requires_grad = False   # they are set to True again in netD update

        netG.zero_grad()

        G_loss = (netD(torch.cat((data_clean, c_data_predict), dim=1)) - 1).pow(2).mean()
        #G_loss = (netD(c_data_predict) - 1).mean().pow(2)

        loss_shadow = loss_function_main(reconstructed = c_data_predict, target = c_data_shadow, original_r = data_predict, original_t = data_shadow).mean()
        
        with torch.no_grad():

          psnr = (10 * torch.log10(1.0 / loss_mse(rgb2yuv(data_predict)[:,0].unsqueeze(1), rgb2yuv(data_shadow)[:,0].unsqueeze(1)).detach())).mean()

          psnrs.append(psnr)
          
          ssim = 1 - instance_ssim(data_predict, data_shadow).mean()
          
          ssims.append(ssim)
          
        loss = loss_shadow + 0.01*(G_loss) 

        loss.backward()

        optimizerG.step()

        if (iteration-1) % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss_shadow:{:.6f} : Psnr:{:.2f} : SSIM:{:2f} : Wasserstein:{} : Lr:{:6f}".format(epoch, iteration, len(training_data_loader), loss_shadow, psnr, ssim, Wasserstein_D.mean(), optimizerG.param_groups[0]['lr']))

            logger.add_scalar('loss', loss.item(), steps)

            logger.add_scalar('psnr', psnr.item(), steps)

            logger.add_scalar('ssim', ssim.item(), steps)

     
        show = []

        if (iteration-1) % 10 == 0:

            for idx, tensor in enumerate(zip(data_clean.data.cpu(), data_mask.data.cpu(), data_shadow.data.cpu(), data_predict.data.cpu().clamp(0,1))):

              if idx >1:

                break

              show.extend([tensor[0], tensor[1], tensor[2], tensor[3]])

            show = torch.stack(show,0)

            show = make_grid(show, nrow = 4, padding = 5)

            logger.add_image('Comparison_nEpochs:{}'.format(epoch), show)

    logger.close() 


def test(test_data_loader, netG):
    psnrs = []
    ssims= []
    
    
    for iteration, batch in enumerate(test_data_loader, 1):

        netG.eval()
        steps = iteration

        data_clean, data_mask, data_shadow = \
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
           
        data_predict = netG(data_clean, data_mask)
          
        with torch.no_grad():

          psnr = (10 * torch.log10(1.0 / loss_mse(rgb2yuv(data_predict)[:,0].unsqueeze(1), rgb2yuv(data_shadow)[:,0].unsqueeze(1)).detach())).mean()
          psnrs.append(psnr)
          
          mse = nn.MSELoss()(rgb2yuv(data_predict)[:,0,:,:], rgb2yuv(data_shadow)[:,0,:,:])
          ssim = 1 - instance_ssim(data_predict, data_shadow).mean().item()
          ssims.append(ssim)
        
    print("PSNR {} SSIM {}".format(sum(psnrs)/len(psnrs), sum(ssims)/len(ssims)))

if __name__ == "__main__":
    os.system('clear')
    main()