from types import SimpleNamespace
import json
import sys
import torch
import os
from torch import nn
from datetime import datetime
import torch.backends.cudnn as cudnn
from timm.models import create_model
from FG_dataloader_mvd import FG_Dataset
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader
import gc
import numpy as np
import psutil

import torchlens as tl
import deepspeed
from deepspeed import DeepSpeedConfig
import matplotlib.pyplot as plt


if '/data/karimike/Documents/ActionRecognition-2024/Models/MVD/mvd' not in sys.path:
    sys.path.append('/data/karimike/Documents/ActionRecognition-2024/Models/MVD/mvd/')
import Models.MVD.mvd.utils as utils
# from Models.MVD.mvd.run_class_finetuning import main
from Models.MVD.mvd import functional as ff
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from timm.data import Mixup
from Models.MVD.mvd.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from Models.MVD.mvd.datasets import build_dataset
from Models.MVD.mvd.engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
from Models.MVD.mvd.utils import NativeScalerWithGradNormCount as NativeScaler
from Models.MVD.mvd.utils import multiple_samples_collate
import utils
import modeling_finetune


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
        
def get_args(mode='finetune'):
    with open('/data/karimike/Documents/ActionRecognition-2024/Models/MVD/mvd/{0}_args.json'.format(mode, 'r')) as json_file:
        args = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
    return args

def random_initialization(net):
    if isinstance(net, nn.Linear) or isinstance(net, nn.BatchNorm2d):
        torch.nn.init.uniform_(net.weight, a=-1.0, b=1.0)
        if torch.is_tensor(net.bias):
            torch.nn.init.uniform_(net.bias, a=-1.0, b=1.0)
        
    if isinstance(net, nn.Conv2d) or isinstance(net, nn.Conv3d):
        torch.nn.init.uniform_(net.weight, a=-1.0, b=1.0)
        
def create_mvd_model(mode, device):
    args = get_args('finetune')
    ds_init = deepspeed.initialize
    if ds_init is not None:
        utils.create_ds_config(args)

    model = create_model(
            args.model,
            pretrained=False,
            img_size=args.input_size,
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_cls_token=args.use_cls_token,
            fc_drop_rate=args.fc_drop_rate,
            use_checkpoint=args.use_checkpoint,
        )

    model = model.to(device)
    if mode == 'pre-trained':
        weights_path = '/data/karimike/Downloads/mvd_b_from_b_ckpt_399.pth'
        checkpoint_model = torch.load(weights_path)
        utils.load_state_dict(model, checkpoint_model['model'], prefix=args.model_prefix)
    elif mode == 'untrained':
        model.apply(random_initialization)
        return model
    else:
        print('Unknown mode')

model_name = 'MVD'
mode = sys.argv[2] #'pre-trained'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {0}: '.format(device))
seg = sys.argv[1]
window = 16
save_batch = 60
batch_size = 3
layer_features = {}
counter = 0
save_trigger = False
start_frame = 0
end_frame = -1
data_generator = get_segment_dataloader(frames_path='/data/karimike/Documents/forrest_study_fmri/DNN/Stimulus/Frames/',
                                        batch_size=batch_size,
                                        start_frame=end_frame+1,
                                        segment=seg, 
                                        window=window)
result_dir = get_result_dir('./Models/{0}'.format(model_name), mode, seg)
model = create_mvd_model(mode=mode, device=device)
model.eval()
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
    local_batch_gpu = local_batch.to(device)
    model_history = tl.log_forward_pass(model, 
                                    local_batch_gpu, 
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

print(datetime.now() - st)    
if layer_features:
    start_frame = end_frame + 1
    end_frame = end_frame + layer_features[module].shape[0]
    save_tensors(start_frame=start_frame, 
                 end_frame=end_frame, 
                 layer_features=layer_features,
                 res_dir=result_dir)