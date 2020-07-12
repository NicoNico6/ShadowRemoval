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
parser.add_argument("--pretrained", default="checkpoints/ShadowRemoval/100_shadowsyns.pth", help="path to folder containing the model")
parser.add_argument("--test", default="./datasets/ISTD_Dataset/ISTD_Dataset/test/", help="path to test dataset")
parser.add_argument("--batchSize", default = 1, type = int, help="training batch")
parser.add_argument("--save_model_freq", default=5, type=int, help="frequency to save model")
parser.add_argument("--cuda", default=True, type=bool, help="frequency to save model")
parser.add_argument("--parallel", action = 'store_true', help = "parallel training")
parser.add_argument("--is_hyper", default=1, type=int, help="use hypercolumn or not")
parser.add_argument("--is_training", default=1, help="training or testing")
parser.add_argument("--continue_training", action="store_true", help="search for checkpoint in the subfolder specified by `task` argument")
parser.add_argument("--lr_g", default = 3e-4, type = float, help="generator learning rate")
parser.add_argument("--lr_d", default = 1e-3, type = float, help = "discrimator learning rate")
parser.add_argument("--epoch", default = 500, type = int, help = "training epoch")
parser.add_argument("--acceleration", default = False, type = bool, help = "activating acceleration or mix precision acceleration")
parser.add_argument("--opt_level", default = "O1", help = "setting the apex mode")
parser.add_argument("--local_rank", default=0, type=int)

      
def main():

    global opt, name, logger, netG, netD, vgg, curriculum_ssim, loss_mse, rgb2yuv, instance_ssim, loss_bce 

    opt = parser.parse_args()
    
    name = "ShadowRemoval"
    
    print(opt)

    # Tag_ResidualBlocks_BatchSize

    cuda = opt.cuda

    if cuda and not torch.cuda.is_available():

        raise Exception("No GPU found, please run without --cuda")

    seed = 1334

    torch.manual_seed(seed)
    
    if 'WORLD_SIZE' in os.environ:
      opt.distributed = int(os.environ['WORLD_SIZE']) > 1
          
    if cuda and not torch.cuda.is_available():

        raise Exception("No GPU found, please run without --cuda")
    
    if opt.parallel:
      opt.gpu = opt.local_rank
      torch.cuda.set_device(opt.gpu)
      torch.distributed.init_process_group(backend='nccl', init_method='env://')
      opt.world_size = torch.distributed.get_world_size()
    
    if cuda:

        torch.cuda.manual_seed(seed)

    cudnn.benchmark = True

    print("==========> Loading datasets")
    test_dataset = DatasetFromFolder(
        opt.test, 
        transform=Compose([ToTensor()]),
        training = False,
        experiments = "ShadowRemoval"
        )


    data_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=False)
    
    print("==========> Building model")
    netG = ShadowRemoval(channels =64)
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
    
    print("==========> Setting GPU")
    if cuda:
        netG = netG.cuda()  
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
          netG, optimizerG = amp.initialize(netG, optimizerG, opt_level=opt.opt_level)
                  
    else:
        netG = netG.cpu()
        
        instance_ssim = instance_ssim.cpu()
        loss_smooth_l1 = loss_smooth_l1.cpu()        
        loss_mse = loss_mse.cpu()
        loss_l1 = loss_l1.cpu()
        loss_bce = loss_bce.cpu()
        curriculum_ssim = curriculum_ssim.cpu()
        loss_perceptual = loss_perceptual.cpu()
        rgb2yuv = rgb2yuv.cpu()
    
    test(data_loader, netG)
            
def test(test_data_loader, netG):
    psnrs_clean = []
    psnrs_mask = []
    
    ssims_clean = []
    ssims_mask= []

    bces_mask = []
    
    if not os.path.exists("datasets/removals/"):
      os.mkdir("datasets/removals/")
        
    if not os.path.exists("datasets/removals/clean/"):
      os.mkdir("datasets/removals/clean/")
        
    if not os.path.exists("datasets/removals/mask/"):
      os.mkdir("datasets/removals/mask")
        
    if not os.path.exists("datasets/removals/shadow/"):
      os.mkdir("datasets/removals/shadow/")
          
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
           
          
        with torch.no_grad():
          predict_clean, predict_mask = netG(data_shadow)
        
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
        
        data_clean = Image.fromarray(np.uint8(torch.cat((data_clean, data_mask.expand_as(data_clean), predict_clean), dim = 3).cpu().data[0].permute(1,2,0).mul(255)))
        data_mask = Image.fromarray(np.uint8(torch.cat((data_mask.expand_as(data_shadow), predict_mask.expand_as(data_shadow)), dim = 3).cpu().data[0].permute(1,2,0).mul(255)))
        data_shadow = Image.fromarray(np.uint8(data_shadow.cpu().data[0].permute(1,2,0).mul(255)))   
        
        data_shadow.save("datasets/removals/shadow/{}_{}_{}.jpg".format(iteration, psnr_clean, ssim_clean))
        data_clean.save("datasets/removals/clean/{}_{}_{}.jpg".format(iteration, psnr_clean, ssim_clean))
        data_mask.save("datasets/removals/mask/{}_{}_{}.jpg".format(iteration, psnr_clean, ssim_clean))
        print("processing {}th".format(iteration))
        
    print("psnr_mask:{} psnr_clean:{} ssim_mask:{} ssim_clean:{} bce_mask:{}".format(sum(psnrs_mask)/len(psnrs_mask), sum(psnrs_clean)/len(psnrs_clean), sum(ssims_mask)/len(ssims_mask), sum(ssims_clean)/len(ssims_clean), sum(bces_mask)/len(bces_mask)))
        
        
        
if __name__ == "__main__":
    os.system('clear')
    main()
