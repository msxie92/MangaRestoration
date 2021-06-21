# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from PIL import Image
import numpy as np
import math
import os
import argparse
import importlib
import datetime
import json

### My libs
from core.utils import set_device, postprocess, set_seed
from core.datasetreal import Dataset
import warnings
warnings.filterwarnings("ignore")
 

parser = argparse.ArgumentParser(description="MGP")
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-n", "--model_name", type=str, required=True)
parser.add_argument("-s", "--size", default=None, type=int)
parser.add_argument("-p", "--port", type=str, default="23451")
parser.add_argument('-sl', '--scale', default=4, type=float)
parser.add_argument('-f', '--fix', action='store_true')
args = parser.parse_args()

BATCH_SIZE = 1

def npad(im, pad=128):
  h,w = im.shape[-2:]
  hp = h //pad*pad+pad
  wp = w //pad*pad+pad
  return F.pad(im, (0, wp-w, 0, hp-h), mode='constant', value=1)

def main_worker(gpu, ngpus_per_node, config):


  torch.cuda.set_device(gpu)
  set_seed(config['seed'])

  # Model and version
  net = importlib.import_module('model.'+args.model_name)
  scalepdt = set_device(net.ScaleEstimator())
  path = os.path.join(config['save_dir'], 'scl_00200.pth')
  data = torch.load(path, map_location = lambda storage, loc: set_device(storage)) 
  scalepdt.load_state_dict(data['netG'])
  scalepdt.eval()
  model = set_device(net.MangaRestorator())
  latest_epoch = open(os.path.join(config['save_dir'], 'latest.ckpt'), 'r').read().splitlines()[-1]
  path = os.path.join(config['save_dir'], 'gen_{}.pth'.format(latest_epoch))
  data = torch.load(path, map_location = lambda storage, loc: set_device(storage)) 
  model.load_state_dict(data['netG'], strict=False)
  model.eval()

  # prepare dataset
  dataset = Dataset(config['data_loader'], debug=False, split='test')
  step = math.ceil(len(dataset) / ngpus_per_node)
  dataset.set_subset(gpu*step, min(gpu*step+step, len(dataset)))
  dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=False, num_workers=config['trainer']['num_workers'], pin_memory=True)

  path = os.path.join(config['save_dir'], 'results_real_{}'.format(str(latest_epoch).zfill(5)))
  os.makedirs(path, exist_ok=True)
  # iteration through datasets
  for idx, (images, names) in enumerate(dataloader):
    print('[{}] GPU{} {}/{}: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      gpu, idx, len(dataloader), names[0]))
    N,C,H,W = images.size()
    if H > 1800:
      continue
    images = set_device(images)
    # print(images.shape)
    images = npad(images)
    with torch.no_grad():
      pred_scl = scalepdt(images).cpu()
      # print(pred_scl)
    if config['data_loader']['fix']:
      pred_scl[:] = config['data_loader']['scl']
    with torch.no_grad():
      pred_imgs, atten = model(images, pred_scl.item())

    outH,outW = round(H*pred_scl.item()), round(W*pred_scl.item())
    pred_imgs = pred_imgs.cpu()[:,:,:outH, :outW]

    predsr_imgs2 = postprocess(pred_imgs)[:,:,:,0]
    for i in range(len(predsr_imgs2)):
      Image.fromarray(predsr_imgs2[i]).save(os.path.join(path, '{}_pred.png'.format(names[i].split('.')[0])))
  print('Finish in {}'.format(path))



if __name__ == '__main__':
  ngpus_per_node = torch.cuda.device_count()
  config = json.load(open(args.config))
  if args.size is not None:
    config['data_loader']['w'] = config['data_loader']['h'] = args.size
    config['data_loader']['scl'] = args.scale
    config['data_loader']['fix'] = args.fix
  config['model_name'] = args.model_name
  config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}'.format(config['model_name'], 
    config['data_loader']['name'], config['data_loader']['mask'], config['data_loader']['w']))

  print('using {} GPUs for testing ... '.format(ngpus_per_node))
  # setup distributed parallel training environments
  ngpus_per_node = torch.cuda.device_count()
  config['world_size'] = ngpus_per_node
  config['init_method'] = 'tcp://127.0.0.1:'+ args.port 
  config['distributed'] = True
  mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
 
