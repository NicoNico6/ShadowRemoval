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

from data import SynsDataset
from models.networks import ShadowMattingNet, Discrimator
from models.perceptual import perceptual
from utils import save_checkpoint, rgb2yuv

from ssim import SSIM, CLBase

parser = argparse.ArgumentParser(description="PyTorch ShadowRemoval")
parser.add_argument("--pretrained", default="checkpoints/ShadowSyns/200_shadowsyns.pth", help="path to folder containing the model")
parser.add_argument("--clean_data", default="./datasets/Shadow_USR/shadow_USR/shadow_free/", help="path to real train dataset")
parser.add_argument("--mask_data", default="./datasets/ISTD_Dataset/ISTD_Dataset/train/train_B/", help="path to test dataset")
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

      
def main():

    global opt, name, logger, netG, netD, vgg, curriculum_ssim, loss_mse, rgb2yuv, instance_ssim 

    opt = parser.parse_args()
    
    name = "ShadowSyns"
    
    print(opt)

    # Tag_ResidualBlocks_BatchSize

    cuda = opt.cuda

    if cuda and not torch.cuda.is_available():

        raise Exception("No GPU found, please run without --cuda")

    seed = 1334

    torch.manual_seed(seed)

    if cuda:

        torch.cuda.manual_seed(seed)

    cudnn.benchmark = True

    print("==========> Loading datasets")
    dataset = SynsDataset(
        opt.clean_data,
        opt.mask_data, 
        transform=Compose([ToTensor()]),
        )


    data_loader = DataLoader(dataset=dataset, num_workers=4, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=True)
    
    print("==========> Building model")
    netG = ShadowMattingNet(channels =64, depth = 9)

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            netG = nn.DataParallel(netG, [0, 1, 2, 3])
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            netG.load_state_dict(weights['state_dict'])
            netG = netG.cuda()  
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("==========> Setting GPU")
    if cuda:
        if opt.parallel:
          netG = nn.DataParallel(netG, [0, 1, 2, 3]).cuda()
          
        else:
          netG = netG.cuda()
    else:
        netG = netG.cpu()
    
    Syns(data_loader, netG)
def Syns(data_loader, netG):
    psnrs = []
    ssims= []
    
    for iteration, batch in enumerate(data_loader, 1):
         
        netG.eval()
        steps = iteration

        syns = []
        
        for item in batch:
          item = Variable(item, requires_grad=False)
          if opt.cuda:
            item = item.cuda()
          else:
            item = item.cpu()
        
        if not os.path.exists("datasets/syns"):
          os.mkdir("datasets/syns")
        
        if not os.path.exists("datasets/syns/clean"):
          os.mkdir("datasets/syns/clean")
        
        if not os.path.exists("datasets/syns/mask"):
          os.mkdir("datasets/syns/mask")
        
        if not os.path.exists("datasets/syns/shadow"):
          os.mkdir("datasets/syns/shadow")
          
        for i in range(1, 4):
            with torch.no_grad():
              data_shadow = Image.fromarray(np.uint8(netG(batch[0], batch[i])[0].permute(1,2,0).mul(255).data.cpu().numpy()))
              data_clean = Image.fromarray(np.uint8(batch[0][0].permute(1,2,0).mul(255).cpu().data.numpy()))
              data_mask = Image.fromarray(np.uint8(batch[i][0].permute(1,2,0).mul(255).data.cpu().numpy()))
            
            data_shadow.save("datasets/syns/shadow/{}_{}.jpg".format(iteration, i))
            data_clean.save("datasets/syns/clean/{}_{}.jpg".format(iteration, i))
            data_mask.save("datasets/syns/mask/{}_{}.jpg".format(iteration, i))
            print("Synsing the {}th Paired Images in datasets/syns/".format(iteration))  
            
        
        
        
if __name__ == "__main__":
    os.system('clear')
    main()