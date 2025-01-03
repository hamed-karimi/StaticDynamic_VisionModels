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
import psutil
from datetime import datetime
import webdataset as wds
import matplotlib.pyplot as plt
from LitModel import LitModel
import lightning.pytorch as pl

def get_parameters():
    with open('./Parameters.json', 'r') as json_file:
        params = json.load(json_file,
                           object_hook=lambda d: SimpleNamespace(**d))
    return params


def train(params, model_id, save_dir, train_dataset, val_dataset, logger):
    
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
    
    train_model = LitModel(frames_dir=params.FRAMES_DIR,
                           model_id=model_id, 
                           lr=torch.tensor(params.LEARNING_RATE), 
                           momentum=torch.tensor(params.MOMENTUM), 
                           weight_decay=torch.tensor(params.WEIGHT_DECAY),
                           lr_steps=params.LEARNING_RATE_STEPS)
    
    trainer.fit(train_model, 
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

def main():
    model_type = sys.argv[1]
    model_id = int(sys.argv[2])
    params = get_parameters()
    
    shards = params.SHARDS_DIR    
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
    
    save_dir = './Model Versions/{0}/{1}'.format('supervised dynamic net', model_id)
    if not pexists(save_dir):
        os.makedirs(save_dir)
    logger = TensorBoardLogger(save_dir, name='logs')
    train(params, model_id, save_dir, train_dataset, val_dataset, logger)
    
if __name__ == "__main__":
    main()