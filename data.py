import os
from os import listdir
from os.path import join
from os.path import basename

import torch.utils.data as data

import torchvision.transforms as transforms
import cv2
from PIL import Image
import random

def is_image_file(filename):

  filename_lower = filename.lower()

  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

class DatasetFromFolder(data.Dataset):

    def __init__(self, data_dir, training = True, transform=None, experiments = "ShadowSyns"):

        super(DatasetFromFolder, self).__init__()
        self.transform = transform

        self.training = training
        
        self.clean_filenames, self.mask_filenames, self.shadow_filenames = self.generate_filenames(data_dir) 

        
        assert experiments in ["ShadowRemoval", "ShadowSyns"]
        
        self.experiments = experiments
    
    def listdir(dir):
        names = []
        roots = []
        for root, dirs, files in os.walk(dir):
          names.append(files)
          roots.append(root)
        
        with open("ISDT.txt", 'w') as file:
          for name in names[-1]:
              file.write('{} {} {} \n'.format(os.path.join(roots[1], name), os.path.join(roots[2], name), os.path.join(roots[3], name)))
    
    def generate_filenames(self, data_dir):

        clean = []
        mask = []
        shadow = []
        
        names = []
        roots = []
        
        for root, dirs, files in os.walk(data_dir):
          names.append(files)
          roots.append(root)
        #print(roots[1:])
        for name in names[-1]:
          if self.training:
            clean.append(os.path.join(roots[1], name))
            shadow.append(os.path.join(roots[2], name))
            mask.append(os.path.join(roots[3], name))
          else:
            mask.append(os.path.join(roots[1], name))
            shadow.append(os.path.join(roots[2], name))
            clean.append(os.path.join(roots[3], name))
        return clean, mask, shadow
        
    def RandomResizedCropRotate(self, clean, mask, shadow, th, tw):

        w, h = clean.size

        m = random.randint(0, 2)

        clean, mask, shadow = clean.resize((int(w*(1 + m*0.1)),int(h*(1 + m*0.1))), Image.ANTIALIAS), mask.resize((int(w*(1 + m*0.1)),int(h*(1 + m*0.1))), Image.ANTIALIAS), shadow.resize((int(w*(1 + m*0.1)),int(h*(1 + m*0.1))), Image.ANTIALIAS)  

        
        assert (w>=tw and h>=th), 'w:{} < tw:{} or h:{}<th:{}'.format(w,tw,h,th)

        i = random.randint(0, h-th)

        j = random.randint(0, w-tw)

        croped_clean, croped_mask, croped_shadow = clean.crop((j, i, j+tw, i+th)), mask.crop((j, i, j+tw, i+th)), shadow.crop((j, i, j+tw, i+th)),

        k = random.randint(0, 3)

        return croped_clean.rotate(k*90), croped_mask.rotate(k*90), croped_shadow.rotate(k*90)

    def __getitem__(self, index):

        clean = Image.fromarray(cv2.cvtColor(cv2.imread(self.clean_filenames[index]), cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(cv2.cvtColor(cv2.imread(self.mask_filenames[index]), cv2.COLOR_BGR2RGB))
        shadow = Image.fromarray(cv2.cvtColor(cv2.imread(self.shadow_filenames[index]), cv2.COLOR_BGR2RGB))

        if self.transform:

            if self.training:

              #clean, mask, shadow = (self.RandomResizedCropRotate(clean, mask, shadow, 256, 256))
              clean = clean.resize((320, 320), Image.ANTIALIAS)
              mask = mask.resize((320, 320), Image.ANTIALIAS)
              shadow = shadow.resize((320, 320), Image.ANTIALIAS)
            else:
              clean = clean.resize((320, 320), Image.ANTIALIAS)
              mask = mask.resize((320, 320), Image.ANTIALIAS)
              shadow = shadow.resize((320, 320), Image.ANTIALIAS)
              
            clean = self.transform(clean)
            mask = self.transform(mask)[0,:,:].unsqueeze(0)
            shadow = self.transform(shadow)
            
        if self.experiments == "ShadowRemoval":
          return shadow, mask, clean
        else:
          return clean, mask, shadow

    def __len__(self):

        return len(self.clean_filenames)
        

class SynsDataset(data.Dataset):

    def __init__(self, clean_dir, mask_dir, transform=None, combination = 3):

        super(SynsDataset, self).__init__()
        self.combination = 3
        
        self.transform = transform

        self.clean_filenames = self.generate_filenames(clean_dir)
        self.mask_filenames = self.generate_filenames(mask_dir)
        
    def listdir(dir):
        names = []
        roots = []
        for root, dirs, files in os.walk(dir):
          names.append(files)
          roots.append(root)
        
        with open("ISDT.txt", 'w') as file:
          for name in names[-1]:
              file.write('{} {} {} \n'.format(os.path.join(roots[1], name), os.path.join(roots[2], name), os.path.join(roots[3], name)))
    
    def generate_filenames(self, data_dir):
        data = []
        
        names = []
        roots = []
        
        for root, dirs, files in os.walk(data_dir):
          names.append(files)
          roots.append(root)
        for name in names[-1]:
          data.append(os.path.join(roots[0], name))
          
        return data
        
    def RandomResizedCropRotate(self, clean, mask, shadow, th, tw):

        w, h = clean.size

        m = random.randint(0, 2)

        clean, mask, shadow = clean.resize((int(w*(1 + m*0.1)),int(h*(1 + m*0.1))), Image.ANTIALIAS), mask.resize((int(w*(1 + m*0.1)),int(h*(1 + m*0.1))), Image.ANTIALIAS), shadow.resize((int(w*(1 + m*0.1)),int(h*(1 + m*0.1))), Image.ANTIALIAS)  

        
        assert (w>=tw and h>=th), 'w:{} < tw:{} or h:{}<th:{}'.format(w,tw,h,th)

        i = random.randint(0, h-th)

        j = random.randint(0, w-tw)

        croped_clean, croped_mask, croped_shadow = clean.crop((j, i, j+tw, i+th)), mask.crop((j, i, j+tw, i+th)), shadow.crop((j, i, j+tw, i+th)),

        k = random.randint(0, 3)

        return croped_clean.rotate(k*90), croped_mask.rotate(k*90), croped_shadow.rotate(k*90)

    def __getitem__(self, index):

        clean = Image.fromarray(cv2.cvtColor(cv2.imread(self.clean_filenames[index]), cv2.COLOR_BGR2RGB))
        masks = [Image.fromarray(cv2.cvtColor(cv2.imread(self.mask_filenames[random.randint(0, len(self.mask_filenames)-1)]), cv2.COLOR_BGR2RGB)) for i in range(self.combination)]
        
        w, h = clean.size
        
        if self.transform:
            clean = self.transform(clean)
            for i in range(len(masks)):
              masks[i] =self.transform(masks[i].resize((w, h), Image.ANTIALIAS)) 
        
        return [clean] + masks

    def __len__(self):

        return len(self.clean_filenames)
