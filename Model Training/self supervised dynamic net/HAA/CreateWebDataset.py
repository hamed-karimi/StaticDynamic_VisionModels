import torch.nn as nn
import torch
import math
import sys
import math
from PIL import Image
from itertools import islice
import os
from os.path import join as pjoin
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import psutil
from datetime import datetime
import webdataset as wds


def get_parameters():
    with open('./Parameters.json', 'r') as json_file:
        params = json.load(json_file,
                           object_hook=lambda d: SimpleNamespace(**d))
    return params
    
def get_transform():
    transform = transforms.Compose([transforms.CenterCrop(720),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor()])
    return transform

def get_sample(root_dir, usage, window=1, train=0.8):
    # root_dir = '/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/datasets/HAA500_frames/'
    transform = get_transform()
    all_dir_names = [filename for filename in os.listdir(root_dir) if filename[-1].isdigit()]
    if usage == 'Train':
        dir_names = all_dir_names[:math.ceil(len(all_dir_names)*train)]
        print(len(dir_names))
    elif usage == 'Test':
        dir_names = all_dir_names[math.ceil(len(all_dir_names)*train):]
    for dir_name in dir_names:
        print(dir_name)
        all_dir_frames = os.listdir(pjoin(root_dir, dir_name))
        frame_num = len([at_frame for at_frame in all_dir_frames if at_frame.endswith('.jpg')])
        for i in range(frame_num-window+1):
            image_window = []
            key = os.path.splitext(pjoin(dir_name, 'frame_{0:03d}.jpg'.format(i)))[0]
            for j in range(window):
                at_frame = 'frame_{0:03d}.jpg'.format(i+j)
                frame_path = pjoin(root_dir, dir_name, at_frame)
                image = transform(Image.open(frame_path))
                image_window.append(image)
            image_window_tensor = torch.stack(image_window, dim=1)
            sample = {
                '__key__': key,
                'pyd': image_window_tensor,
            }
            
            yield sample

params = get_parameters()
root_dir = params.FRAMES_DIR
output_dir = params.SHARDS
with wds.ShardWriter(pjoin(output_dir, 'out-%06d.tar'), maxcount=10000) as sink:
    for sample in get_sample(root_dir, usage='Train', window=11, train=1):
        sink.write(sample)