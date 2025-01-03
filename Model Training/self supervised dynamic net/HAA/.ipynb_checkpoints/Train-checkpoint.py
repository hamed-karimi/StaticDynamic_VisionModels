import torch.nn as nn
import torch
import math
import sys
import math
from itertools import islice
import os
from os.path import join as pjoin
from os.path import exists as pexists
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from lightning.pytorch.loggers import TensorBoardLogger
from types import SimpleNamespace
import json

if '/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/models/' not in sys.path:
    sys.path.append('/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/models/')
if '/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/' not in sys.path:
    sys.path.append('/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/')

import psutil
from datetime import datetime
import webdataset as wds
import matplotlib.pyplot as plt
from TinyMotionNet_LitModel import LitModel
import lightning.pytorch as pl

def get_parameters(model_type):
    with open('./{0}/Parameters.json'.format(model_type), 'r') as json_file:
        params = json.load(json_file,
                           object_hook=lambda d: SimpleNamespace(**d))
    return params

# def get_transform():
#     new_length = 1
#     scale_ratios = [1.0, 0.875, 0.75, 0.66]
#     clip_mean = [0.485, 0.456, 0.406] * new_length
#     clip_std = [0.229, 0.224, 0.225] * new_length

#     normalize = transforms.Normalize(mean=clip_mean,
#                                      std=clip_std)
#     transform = transforms.Compose([
#             transforms.Resize((256)),
#             transforms.RandomResizedCrop((224, 224), (scale_ratios[1], scale_ratios[0]), (scale_ratios[3], scale_ratios[2])),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ])
#     return transform

def train(params, save_dir, train_dataset, val_dataset, logger):
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=None,
                              num_workers=params.WORKER_NUM,
                              pin_memory=True)
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=None,
                            num_workers=params.WORKER_NUM,
                            pin_memory=True)
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6'
    
    
    trainer = pl.Trainer(accumulate_grad_batches=params.ACCUMULATE_GRAD_BATCHES,
                         default_root_dir=save_dir,
                         max_epochs=params.EPOCH_NUM, 
                         num_nodes=params.GPU_NUM, 
                         strategy ='ddp', 
                         logger=logger)
    
    train_model = LitModel(seg_0, lr=torch.tensor(params.LEARNING_RATE), 
                           momentum=torch.tensor(params.MOMENTUM), 
                           weight_decay=torch.tensor(params.WEIGHT_DECAY),
                           lr_steps=params.LEARNING_RATE_STEPS)
    
    trainer.fit(train_model, 
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

def main():
    model_type = sys.argv[1]
    model_id = int(sys.argv[2])
    params = get_parameters(model_type)
    
    shards = '/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/Shards_TinyMotionNet'
    
    # Images have already been transformed  during creation
    
    train_dataset = (
        wds.WebDataset(shards + '/out-{000000..000873}.tar', shardshuffle=True)
        .decode('pil')
        .to_tuple('__key__', 'pyd')
        .shuffle(1000)
        .batched(params.BATCH_SIZE)
    )
    val_dataset = (
        wds.WebDataset(shards + '/out-{000874..000911}.tar', shardshuffle=True)
        .decode('pil')
        .to_tuple('__key__', 'pyd')
        .shuffle(1000)
        .batched(params.BATCH_SIZE)
    )
    
    save_dir = '/data/karimike/Documents/forrest_study_fmri/Analysis/All Runs/All models/Model Versions/{0}/{1}'.format(model_type, model_id)
    if not pexists(save_dir):
        os.makedirs(save_dir)
    logger = TensorBoardLogger(save_dir, name='logs')
    train(params, save_dir, train_dataset, val_dataset, logger)
    
if __name__ == "__main__":
    main()