import sys
import os
import torch
import json
from types import SimpleNamespace
import cv2
from datetime import datetime
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, VideoMAEForPreTraining, VideoMAEConfig, VideoMAEModel
import numpy as np
import psutil
import time
import gc
from torch import nn
import torchvision
import torchlens as tl
import visualpriors

from torch.utils.data import BatchSampler, SequentialSampler, DataLoader
from FG_dataloader import FG_Dataset
# if '/data/karimike/Documents/forrest_study_fmri/DNN/Stimulus/' not in sys.path:
#     sys.path.append('/data/karimike/Documents/forrest_study_fmri/DNN/Stimulus/')

def get_segment_dataloader(frames_path, batch_size, start_frame, segment, window):
    # from FG_dataloader import FG_Dataset
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
    if mode == 'pre-trained':
        model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
    elif mode == 'fine-tuned':
        VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    elif mode == 'untrained':
        configuration = VideoMAEConfig()
        # # Randomly initializing a model from the configuration
        model = VideoMAEModel(configuration)
    else:
        print('Unknown mode')
        
    return model.to(device)

model_name = 'VideoMAE'
mode = 'untrained' #'pre-trained'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {0}: '.format(device))
seg = int(sys.argv[1])
window = 16
save_batch = 1000
batch_size = 8
layer_features = {}
counter = 0
save_trigger = False
start_frame = 0
end_frame = -1 if len(sys.argv) < 3 else int(sys.argv[2])

data_generator = get_segment_dataloader(frames_path='/data/karimike/Documents/forrest_study_fmri/DNN/Stimulus/Frames/',
                                        batch_size=batch_size,
                                        start_frame=end_frame+1,
                                        segment=seg, 
                                        window=window)


model = create_model(mode=mode, device=device)
num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
sequence_len = (window // model.config.tubelet_size) * num_patches_per_frame
model.eval()
result_dir = get_result_dir('./Models/{0}'.format(model_name), mode, seg)
print(result_dir)

modules_of_interest = [
    'encoder.layer.0.output',
    'encoder.layer.1.output',
    'encoder.layer.2.output',
    'encoder.layer.3.output',
    'encoder.layer.4.output',
    'encoder.layer.5.output',
    'encoder.layer.6.output',
    'encoder.layer.7.output',
    'encoder.layer.8.output',
    'encoder.layer.9.output',
    'encoder.layer.10.output',
    'encoder.layer.11.output',
    # 'fc_norm'
]
if mode in['fine-tuned', 'pre-trained']:
    modules_of_interest = ['videomae.'+module for module in modules_of_interest]
    

st = datetime.now()
for local_batch in data_generator:
    local_batch_gpu = local_batch.to(device)
    # remove masked_tokens when fine-tuning:
    if mode in ['fine-tuned', 'untrained']:
        input_args = local_batch_gpu
    elif mode in ['pre-trained']:
        masked_tokens = torch.zeros((local_batch_gpu.shape[0], sequence_len)).bool()
        masked_tokens[:, 0] = True
        input_args = [local_batch_gpu, masked_tokens]
        
    model_history = tl.log_forward_pass(model=model, 
                                        input_args=input_args, 
                                        vis_opt='none', 
                                        keep_unsaved_layers=False,
                                        # layers_to_save=modules_of_interest,
                                        detach_saved_tensors=True,
                                        output_device='cpu')
    
    layer_i = -1 # last layer which is the add
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