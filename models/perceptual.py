import torch
import torch.nn as nn
import torchvision.models as models

class perceptual(nn.Module):
    def __init__(self, backbones = 'vgg16'):
      super(perceptual, self).__init__()
      
      if backbones == 'vgg16':
        modules = (models.vgg16(pretrained = True).features[:-1])
    
        self.block1 = modules[0:4]
        self.block2 = modules[4:9]
        self.block3 = modules[9:16]
        self.block4 = modules[16:23]
        self.block5 = modules[23:]
      
        for param in self.parameters():
            param.requires_grad = False
      
      else:
        raise ValueError
        
      self.mse = torch.nn.L1Loss()
      self.weights = [2.6, 4.8, 3.7, 5.6, 0.15]
      
    def extraction(self, x):
        out = []
        out.append(self.block1(x))
        out.append(self.block2(out[-1]))
        out.append(self.block3(out[-1]))
        out.append(self.block4(out[-1]))
        out.append(self.block5(out[-1]))
        
        return out
            
    def forward(self, x, y):
        N, C, H, W = x.size()
        
        out_x = self.extraction(x)
        out_y = self.extraction(y)
        
        out = 0
        
        for i in range(len(out_x)):
          out += self.mse(out_x[i], out_y[i]).mul(1/self.weights[i])
            
        return out