import sys
import os
import torch
import json
from types import SimpleNamespace
import cv2
from datetime import datetime
import torchvision.transforms as transforms
from PIL import Image
# import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, ViTMAEConfig, ViTMAEModel #ViTForImageClassification
import numpy as np
import psutil
import time
import gc
from torch import nn
import torchvision
import torchlens as tl
import visualpriors

from torch.utils.data import BatchSampler, SequentialSampler, DataLoader
from FG_dataloader_mae import FG_Dataset
sys.path.append('/data/karimike/Documents/ActionRecognition-2024/Models/ImageMAE/mae')
import models_vit, models_mae

def get_segment_dataloader(frames_path, batch_size, start_frame, segment, window):
    dataset = FG_Dataset(frames_path, segment, window=window)

    start_frame = start_frame #0 #22501
    end_frame = len(dataset)
    print(start_frame, end_frame)
    batch_size = batch_size
    batch_sampler = BatchSampler(SequentialSampler(range(start_frame, end_frame)),
                                 batch_size=batch_size,
                                 drop_last=False)
    params = {'batch_sampler': batch_sampler,
              'pin_memory': True}

    data_generator = DataLoader(dataset, **params)
    
    return data_generator

def get_result_dir(root, mode, segment):
    result_dir = os.path.join(root, mode, 'seg_{0}'.format(segment))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def save_tensors(start_frame, end_frame, layer_features, res_dir):
    for module in layer_features.keys():
        filename = '{0}_{1}_{2}.pt'.format(start_frame, end_frame, module.replace('.', '_'))
        path = os.path.join(res_dir, filename)
        torch.save(layer_features[module], path)
        
def create_model(mode, device):
    model_dir = os.path.join('/data/karimike/Documents/ActionRecognition-2024/Models/ImageMAE', mode)
    if mode == 'pre-trained':
        model = models_mae.__dict__['mae_vit_base_patch16']()
        pretrain_ckpt = '{0}/model/mae_pretrain_vit_base.pth'.format(model_dir)
        weights = torch.load(pretrain_ckpt)
        model.load_state_dict(weights['model'], strict=False)
        
    elif mode == 'fine-tuned':
        model = models_vit.__dict__['vit_base_patch16'](num_classes=1000, 
                                                drop_path_rate=0.1, 
                                                global_pool=True)
        finetuned_ckpt = '{0}/model/mae_finetuned_vit_base.pth'.format(model_dir)
        weights = torch.load(finetuned_ckpt, weights_only=True)
        model.load_state_dict(weights['model'], strict=False)
        
    elif mode == 'untrained':
        configuration = ViTMAEConfig()
        # # Randomly initializing a model from the configuration
        model = ViTMAEModel(configuration)
    else:
        print('Unknown mode')
        
    return model.to(device)

model_name = 'ImageMAE'
mode = 'pre-trained'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {0}: '.format(device))
seg = int(sys.argv[1])
window = 1
save_batch = 1000
batch_size = 100 #50
layer_features = {}
counter = 0
save_trigger = False
start_frame = 0
end_frame = -1 if len(sys.argv) < 3 else int(sys.argv[2]) #-1

data_generator = get_segment_dataloader(frames_path='/data/karimike/Documents/forrest_study_fmri/DNN/Stimulus/Frames/',
                                        batch_size=batch_size,
                                        start_frame=end_frame+1,
                                        segment=seg, 
                                        window=window)

model = create_model(mode=mode, device=device)
processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base', use_fast=True)
model.eval()

result_dir = get_result_dir('./Models/{0}'.format(model_name), mode, seg)
print(result_dir)

modules_of_interest = ['blocks.0.mlp.fc2',
                       'blocks.1.mlp.fc2',
                       'blocks.2.mlp.fc2',
                       'blocks.3.mlp.fc2',
                       'blocks.4.mlp.fc2',
                       'blocks.5.mlp.fc2',
                       'blocks.6.mlp.fc2',
                       'blocks.7.mlp.fc2',
                       'blocks.8.mlp.fc2',
                       'blocks.9.mlp.fc2',
                       'blocks.10.mlp.fc2',
                       'blocks.11.mlp.fc2']

st = datetime.now()
for local_batch in data_generator:
    local_batch_gpu = processor(local_batch)['pixel_values'].to(device)
    model_history = tl.log_forward_pass(model, 
                                    local_batch_gpu, 
                                    vis_opt='none', 
                                    keep_unsaved_layers=False,
                                    # layers_to_save=modules_of_interest,
                                    detach_saved_tensors=True,
                                    output_device='cpu')

    layer_i = -1 # last layer which is the fc2
    for module in modules_of_interest:
        layer_name = model_history.module_layers[module][layer_i]
        layer_tensor = model_history.layer_dict_main_keys[layer_name].tensor_contents
        if module in layer_features and layer_tensor.dim() < layer_features[module].dim():
            layer_tensor = layer_tensor.unsqueeze(0)
            save_trigger = True
        if module in layer_features:
            layer_features[module] = torch.cat([layer_features[module], layer_tensor], dim=0)
        else:
            layer_features[module] = layer_tensor
        
        if layer_features[module].shape[0] % save_batch == 0:
            save_trigger = True
            
    if save_trigger:
        start_frame = end_frame + 1
        end_frame = end_frame + layer_features[module].shape[0]
        save_tensors(start_frame=start_frame, 
                     end_frame=end_frame, 
                     layer_features=layer_features,
                     res_dir=result_dir)
        layer_features = {}
        save_trigger = False
        print(datetime.now() - st)
    
    print(counter)
    counter += local_batch.shape[0]
    del model_history
    gc.collect()
    torch.cuda.empty_cache()
    
if layer_features:
    start_frame = end_frame + 1
    end_frame = end_frame + layer_features[module].shape[0]
    save_tensors(start_frame=start_frame, 
                 end_frame=end_frame, 
                 layer_features=layer_features,
                 res_dir=result_dir)