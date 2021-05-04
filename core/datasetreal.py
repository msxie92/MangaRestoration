import random
from random import shuffle
import os 
import math 
import numpy as np 
from PIL import Image, ImageFilter
from glob import glob

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


from core.utils import ZipReader


class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_args, debug=False, split='train', level=None):
    super(Dataset, self).__init__()
    self.split = split
    self.level = level
    # self.w, self.h = 512,512
    self.w, self.h = data_args['w'], data_args['h']
    self.data = [os.path.join(data_args['zip_root'], data_args['name2'], i) 
      for i in np.genfromtxt(os.path.join(data_args['flist_root'], data_args['name2'], split+'.flist'), dtype=np.str, encoding='utf-8',delimiter="\n")]
      # for i in np.genfromtxt(os.path.join(data_args['flist_root'], data_args['name'], split+'4.flist'), dtype=np.str, encoding='utf-8',delimiter="\n")]
    self.data.sort()
    
    if split == 'train':
      self.data = self.data*data_args['extend']
      shuffle(self.data)
    if debug:
      self.data = self.data[:100]

  def __len__(self):
    return len(self.data)
  
  def set_subset(self, start, end):
    self.data = self.data[start:end] 

  def __getitem__(self, index):
    try:
      item = self.load_item(index)
    except:
      print('loading error: ' + self.data[index])
      item = self.load_item(0)
    return item

  def load_item(self, index):
    # load image
    fname = self.data[index].split(',')[0]
    img_path = os.path.dirname(fname)
    img_name = os.path.basename(fname)
    img = Image.open(os.path.join(img_path, img_name)).convert('L')
    return F.to_tensor(img)*2-1., img_name

  def create_iterator(self, batch_size):
    while True:
      sample_loader = DataLoader(dataset=self,batch_size=batch_size,drop_last=True)
      for item in sample_loader:
        yield item

def get_pos(size, crop_size):
    w, h = size
    new_h = h
    new_w = w
    crop = 0
    x = random.randint(crop, np.maximum(0, new_w - crop_size-crop))
    y = random.randint(crop, np.maximum(0, new_h - crop_size-crop))
    return x, y


def crop(img, size):
    ow, oh = img.size
    x1, y1 = get_pos(img.size, ow)
    tw, th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img