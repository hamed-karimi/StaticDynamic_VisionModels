import torch
from torch import nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import torchlens as tl
from datetime import datetime
import sys
import gc
import psutil
import os
from collections import OrderedDict
from copy import deepcopy

from torch.utils.data import BatchSampler, SequentialSampler, DataLoader
from FG_dataloader_flowclassifier import FG_Dataset

if '/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/' not in sys.path:
    sys.path.append('/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/')
if '/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/MotionNet' not in sys.path:
    sys.path.append('/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/MotionNet')

import models
import feafa_architecture

def get_tinyMotionNet_weights(device):
    snapshot_path = '/data/karimike/Documents/ActionRecognition-2024/Models/TinyMotionNet/Kinetics-trained/Training/snapshot.pt'
    checkpoints = torch.load(snapshot_path, device)
    weights = OrderedDict()
    for key in checkpoints['MODEL_STATE']:
        new_key = deepcopy(key)
        if key.startswith('module'):
            new_key = key[7:]
        weights[new_key] = checkpoints['MODEL_STATE'][key]
    return weights


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
        
model_name = 'TinyMotionNet'
seg = int(sys.argv[1])
mode = sys.argv[2] #'Kinetics-trained'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {0}: '.format(device))

window = 11
save_batch = 600
batch_size = 150 #50
layer_features = {}
counter = 0
save_trigger = False
start_frame = 0
end_frame = -1 if len(sys.argv) < 4 else int(sys.argv[3]) #-1

data_generator = get_segment_dataloader(frames_path='/data/karimike/Documents/forrest_study_fmri/DNN/Stimulus/Frames/',
                                        batch_size=batch_size,
                                        start_frame=end_frame+1,
                                        segment=seg, 
                                        window=window)

model = feafa_architecture.TinyMotionNet().to(device)
weights = get_tinyMotionNet_weights(device=device)
model.load_state_dict(weights)
model.eval()

result_dir = get_result_dir('./Models/{0}'.format(model_name), mode, seg)
print(result_dir)

modules_of_interest = ['conv1', 
                       'conv2', 
                       'conv3', 
                       'conv4', 
                       'predict_flow4', 
                       'deconv3', 
                       'xconv3', 
                       'predict_flow3',     
                       'deconv2', 
                       'xconv2', 
                       'predict_flow2']

st = datetime.now()
for local_batch in data_generator:
    local_batch_gpu = local_batch.to(device)
    model_history = tl.log_forward_pass(model, 
                                    local_batch_gpu, 
                                    vis_opt='none', 
                                    keep_unsaved_layers=False,
                                    # layers_to_save=modules_of_interest,
                                    detach_saved_tensors=True,
                                    output_device='cpu')
    
    layer_i = -1 # last layer which is the conv2d
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