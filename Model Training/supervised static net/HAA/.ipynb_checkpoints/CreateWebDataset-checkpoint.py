import torch.nn as nn
import torch
import math
import json
from types import SimpleNamespace
import sys
from itertools import islice
import os
from os.path import join as pjoin
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import webdataset as wds

def get_parameters():
    with open('./Parameters.json', 'r') as json_file:
        params = json.load(json_file,
                           object_hook=lambda d: SimpleNamespace(**d))
    return params
    
def get_labels(frames_dir):
    all_frames = [frame[0:-4] for frame in os.listdir(frames_dir) if frame[-1].isdigit()]
    unique_frame_names = set(all_frames)
    label_indices = {}
    for i, label in enumerate(unique_frame_names):
        label_indices[label] = i
    return label_indices

def get_sample(root_dir, usage, train=0.8):
    count = 0
    all_dir_names = os.listdir(root_dir)
    if usage == 'Train':
        dir_names = all_dir_names[:math.ceil(len(all_dir_names)*train)]
    elif usage == 'Test':
        dir_names = all_dir_names[math.ceil(len(all_dir_names)*train):]
    all_labels = get_labels(root_dir)
    for dir_name in dir_names:
        if not dir_name[0:-4] in all_labels:
            continue
        sample_class = all_labels[dir_name[0:-4]]
        dir_frames = os.listdir(pjoin(root_dir, dir_name))
        for at_frame in dir_frames:
            if not at_frame.endswith('.jpg'):
                continue
            frame_path = pjoin(root_dir, dir_name, at_frame)
            with open(frame_path, 'rb') as stream:
                binary_data = stream.read()
                key = os.path.splitext(pjoin(dir_name, at_frame))[0]

                sample = {
                    '__key__': key,
                    'jpg': binary_data,
                    'cls': sample_class
                }

                yield sample

params = get_parameters()
root_dir = params.FRAMES_DIR
output_dir = params.SHARDS
with wds.ShardWriter(pjoin(output_dir, 'out-%04d.tar'), maxcount=10000) as sink:
    for sample in get_sample(root_dir, usage='Train', train=1):
        sink.write(sample)