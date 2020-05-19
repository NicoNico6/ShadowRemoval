# coding=utf-8

# author: guonianhui199512@gmail.com

# version: 0.1.0

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

from math import exp



def gaussian(window_size, sigma):

    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])

    return gauss/gauss.sum()



def create_window(window_size, channel):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)

    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window



def _ssim(img1, img2, window, window_size, channel, L=1, reduction='mean'):

        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)

        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)



        mu1_sq = mu1.pow(2)

        mu2_sq = mu2.pow(2)

        mu1_mu2 = mu1*mu2



        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq

        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq

        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2



        C1 = (0.01 * L)**2

        C2 = (0.03 * L)**2



        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    

        if reduction == 'mean':

            return ssim_map.mean()

        elif reduction == 'sum':

            return ssim_map.sum()

        elif reduction == 'none':

            return ssim_map

        raise ValueError(reduction + " is not a valid value for reduction")



class SSIM(torch.nn.Module):

    def __init__(self, window_size = 11, L=1, reduction='mean', asloss=True):

        super(SSIM, self).__init__()

        self.window_size = window_size

        self.channel = 1

        self.window = create_window(window_size, self.channel)

        self.L = L

        self.reduction = reduction

        self.asloss = asloss

    

    def _ssim(self, img1, img2, window, window_size, channel, L=1, reduction='mean'):

        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)

        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)



        mu1_sq = mu1.pow(2)

        mu2_sq = mu2.pow(2)

        mu1_mu2 = mu1*mu2



        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq

        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq

        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2



        C1 = (0.01 * L)**2

        C2 = (0.03 * L)**2



        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    

        if reduction == 'mean':

            return ssim_map.mean()

        elif reduction == 'sum':

            return ssim_map.sum()

        elif reduction == 'none':

            return ssim_map

        raise ValueError(reduction + " is not a valid value for reduction")

    

    def forward(self, img1, img2):

        (_, channel, _, _) = img1.size()



        if channel == self.channel and self.window.data.type() == img1.data.type():

            window = self.window

        else:

            window = create_window(self.window_size, channel)

            

            if img1.is_cuda:

                window = window.cuda(img1.get_device())

            window = window.type_as(img1)

            

            self.window = window

            self.channel = channel

            

        ssim = self._ssim(img1, img2, window, self.window_size, channel, L=self.L, reduction=self.reduction)

        return 1 - ssim if self.asloss else ssim



def ssim(img1, img2, window_size = 11, L=1, reduction='mean'):

    (_, channel, _, _) = img1.size()

    window = create_window(window_size, channel)

    

    if img1.is_cuda:

        window = window.cuda(img1.get_device())

    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, L=L, reduction=reduction)



class CLBase(nn.Module):

    def __init__(self, maxv=2, alpha=0.2, keep=True, momentum=0., lossfunc = SSIM(reduction='none', window_size=7, asloss = False)):

        super(CLBase, self).__init__()

        self.keep = keep
        
        self.maxv = maxv

        self.alpha = alpha

        self.lossfunc = lossfunc

        self.momentum = momentum
        
        self.running_mean = None

    def forward(self, deblur, clear, age):
        sigma2p1 = self.alpha * age

        sigma2n1 = sigma2p1 / 1
        if self.running_mean is None:
          self.running_mean = torch.ones(1, deblur.size(1)) if self.keep else None
          self.running_mean.mul_(-1)
          
        
        if deblur.is_cuda and self.running_mean is not None:

          self.running_mean = self.running_mean.cuda(deblur.get_device())

        with torch.no_grad():

            lossmaps = self.lossfunc(deblur, clear)
            lossavg = lossmaps.mean(dim=-1).mean(dim=-1)

        if self.running_mean is not None:

            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * lossavg.mean(dim=0)[None, :]
            mu = self.running_mean.repeat(lossmaps.size(0), 1)[:, :, None, None]
            
        else:

            # instance level

            mu = lossavg.mean(dim=-1)

            mu = mu[:, None, None, None]

        with torch.no_grad():

            gauss = lambda x, sigma: torch.exp(-((x-mu) / sigma) ** 2) * self.maxv

            weight_p = gauss(lossmaps, sigma2p1).detach()

            weight_n = gauss(lossmaps, sigma2n1).detach()

            weight = torch.where(lossmaps > mu, weight_p, weight_n)
            deblur_n, clear_n = deblur * weight, clear * weight


        return deblur_n, clear_n, weight