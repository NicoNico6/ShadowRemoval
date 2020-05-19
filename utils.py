import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F

import math
import numpy as np

import os
from os import listdir
from os.path import join

import torchvision.transforms as transforms


def weights_init_kaiming(m):

  classname = m.__class__.__name__

  if classname.find('Conv') != -1:

    nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')

  elif classname.find('Linear') != -1:

    nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')

  elif classname.find('BatchNorm') != -1:

    m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)

    nn.init.constant(m.bias.data, 0.0)

def output_psnr_mse(img_orig, img_out):

  squared_error = np.square(img_orig - img_out)
  mse = np.mean(squared_error)
  psnr = 10 * np.log10(1.0 / mse)
  return psnr

def is_image_file(filename):

  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat', '.jpeg', '.JPEG'])

def load_all_image(path):
  return [join(path, x) for x in listdir(path) if is_image_file(x)]

def save_checkpoint(model, epoch, model_folder):
  model_out_path = "checkpoints/%s/%d_.pth" % (model_folder, epoch)
  state_dict = model.state_dict()
  for key in state_dict.keys():
    state_dict[key] = state_dict[key].cpu()

  if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
    
  if not os.path.exists("checkpoints/" + model_folder):
    os.makedirs("checkpoints/" + model_folder)
    
  torch.save({
    'epoch': epoch,
    'state_dict': state_dict}, model_out_path)
    	
  print("Checkpoint saved to {}".format(model_out_path))

class rgb2yuv(nn.Module):
  def __init__(self, agreement = "BT601"):
    super(rgb2yuv, self).__init__()
    if agreement == "BT601":
       self.wr, self.wg, self.wb = 0.299, 0.587, 0.114
    elif agreement == "BT709":
       self.wr, self.wg, self.wb = 0.2126, 0.7152, 0.0772
    elif agreement == "BT2020":
       self.wr, self.wg, self.wb = 0.2627, 0.678, 0.0593
    else:
        raise ValueError

    self.vmax, self.umax = 0.5, 0.5

  def forward(self, rgb):
    
    r = rgb[:, 0, :, :].unsqueeze(1).clamp(0,1)
    g = rgb[:, 1, :, :].unsqueeze(1).clamp(0,1)
    b = rgb[:, 2, :, :].unsqueeze(1).clamp(0,1)
    
    y = (r.mul(self.wr) + g.mul(self.wg) + b.mul(self.wb)).mul(219.0/255.0).add(16.0/255.0)
    cb = (b - y).mul(self.umax).div( 1 - self.wb).mul(224.0/255.0).add(128.0/255.0)
    cr = (r - y).mul(self.vmax).div( 1 - self.wr).mul(224.0/255.0).add(128.0/255.0)
    
    return  torch.cat((y,cb,cr), 1)

class yuvloss(nn.Module):
	def __init__(self):
		super(yuvloss, self).__init__()
		
		self.rgb2yuv = rgb2yuv()
		self.creterion = nn.MSELoss()
	
	def forward(self, input, target):
		return self.creterion(self.rgb2yuv(input), self.rgb2yuv(target)).mean()
		
class sobel(nn.Module):
  def __init__(self, kernel_size = 3):
    super(sobel, self).__init__()
    self.kernel_size = kernel_size
    
    if kernel_size == 3:
      self.sobel_kernel_col = Variable(torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1,1,3,3))
      self.sobel_kernel_row = Variable(torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1,1,3,3))
    elif kernel_size == 5:
      self.sobel_kernel_col = Variable(torch.Tensor([[2, 1, 0, -1, -2], [3, 2, 0, -2, -3], [4, 3, 0, -3, -4], [3, 2, 0, -2, -3], [2, 1, 0, -1, -2]]).view(1,1,5,5))
      self.sobel_kernel_row = Variable(torch.Tensor([[2, 3, 4, 3, 2], [1, 2, 3, 2, 1], [0, 0, 0, 0, 0], [-1, -2, -3, -2, -1], [-2, -3, -4, -3, -2]]).view(1,1,5,5))
    else:
      raise ValueError
    
  def forward(self, x):
    assert x.size(1) == 3
    if x.is_cuda:
        self.sobel_kernel_col = self.sobel_kernel_col.cuda(x.get_device())
        self.sobel_kernel_row = self.sobel_kernel_row.cuda(x.get_device())
    
    self.sobel_kernel_col = self.sobel_kernel_col.type_as(x)    
    self.sobel_kernel_row = self.sobel_kernel_row.type_as(x)
     
    x_gray = (0.299*x[:,0] + x[:,1]*0.587 + 0.114*x[:,2]).unsqueeze(1)
    col = F.conv2d(x_gray, self.sobel_kernel_col, padding = self.kernel_size // 2, groups = 1)
    row = F.conv2d(x_gray, self.sobel_kernel_row, padding = self.kernel_size // 2, groups = 1)
    
    return (col.pow(2) + row.pow(2)).add(1e-20).sqrt(), torch.atan(row.mul(col.add(1e-20).pow(-1)))


    


