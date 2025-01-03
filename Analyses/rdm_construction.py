import numpy as np
import os
import gc
from copy import deepcopy
from nipy.modalities.fmri.hrf import spmt
from datetime import datetime
import nibabel as nib
import sys
import re
import torch
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import itertools
import json
from types import SimpleNamespace
import psutil

def pearson_corr(m1, m2):
    m1_z = zscore(m1, axis=0)
    m2_z = zscore(m2, axis=0)
    
    corr = np.matmul(np.transpose(m1_z), m2_z) / m1_z.shape[0]
    return corr

def get_hrf(t1, t2, fps): #fps = 25/5
    hrf_t1_t2 = spmt(np.arange(t1, t2, 1 / fps))
    return hrf_t1_t2

def batch_names_key(batch_name, batch_split_ind):
    str_segment = re.split('[_]', batch_name)[batch_split_ind]
    return int(str_segment)

def get_segment_activation(model_layers_dir, seg, layer, batch_split_ind): # -> (time, features, ...)
    layers_dir = '{0}/seg_{1}'.format(model_layers_dir, seg)
    layer_batch_names = [name for name in os.listdir(layers_dir) if '_'+layer in name]
    layer_batch_names.sort(key=lambda key: batch_names_key(key, batch_split_ind))
    layer_batch_path = [os.path.join(layers_dir, name) for name in layer_batch_names]
    layer_activation = []

    for batch_name in layer_batch_path:
        at_batch = torch.load(batch_name) #.T
        sampling_rate = 5
        samples = range(0, at_batch.shape[0], sampling_rate)
        layer_activation.append(at_batch[samples, :])
        del at_batch

    layer_activation = torch.concat(layer_activation, dim=0)
    gc.collect()
        
    return layer_activation

def layer_convolution(layer_activation):
    convolved = np.zeros(layer_activation.shape, dtype=np.float32)
    hrf = get_hrf(0, 20, 25/5)
    for n in range(layer_activation.shape[0]):# Convolution -> speareds activation to time+99 
        convolved[n, :] = np.convolve(hrf, layer_activation[n, :])[:layer_activation.shape[1]]
        if n % 100000 == 0:
            gc.collect()
            # print(n)
    gc.collect()
    return convolved

model_name = sys.argv[1] #'VideoMAE'
mode = sys.argv[2]
seg = int(sys.argv[3])
model_dir = os.path.join('/data/karimike/Documents/ActionRecognition-2024/Models', model_name, mode)
params_dir = os.path.join('{0}/Parameters.json'.format(model_dir))
with open(params_dir, 'r') as json_file:
        params = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
model_modules = params.MODULES
ref_subject = 1
do_convolution = True
first_frame = 25
split_index = params.SPLIT_INDEX

for module in model_modules:
    # for seg in range(seg_range):
    layer_activation = get_segment_activation(model_layers_dir=model_dir, 
                                               seg=seg, 
                                               layer=module.replace('.', '_'),
                                               batch_split_ind=split_index).flatten(start_dim=1, 
                                                                                    end_dim=-1)
    fmri_time = nib.load('/data/karimike/Documents/forrest_study_fmri/sub-{0:02d}_complete/func/compcorr/func_run-{1:02d}.nii.gz'.format(ref_subject, seg+1)).shape[3]
    frames_trigger = (np.array([50] * fmri_time).cumsum() - (50 - first_frame)) // (25//5)
    convolved = layer_convolution(layer_activation.T) # convolve each feature over time with the hrf func
    del layer_activation
    rdm = 1 - pearson_corr(convolved[:, frames_trigger], convolved[:, frames_trigger])
    gc.collect()

    if do_convolution:
        result_dir = '{0}/seg_{1}/rdms'.format(model_dir, seg) 
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        np.save('{0}/{1}'.format(result_dir, module), rdm)

    else:
        np.save('{0}/{1}_seg_{2}_seg_{3}_noconv.npy'.format(result_dir, layer, seg1, seg2), rdm)

    del convolved
    del frames_trigger

    print('done: ', module, seg)
    print('memory usage: ', psutil.virtual_memory().used / 1000000000)
    gc.collect()
