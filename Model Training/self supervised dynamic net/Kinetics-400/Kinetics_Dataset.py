import gc
from datetime import datetime
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import psutil
import numpy as np
import pickle
import cv2
from torch.utils.data.dataset import T_co

class KDataset(Dataset):

    def __init__(self, params, usage, window, step_size):
        self.window = window
        self.step_size = step_size
        self.transform = transforms.Compose(
            [transforms.CenterCrop(720), transforms.Resize((224, 224)), transforms.ToTensor()])
        # self.all_frames = self.load_all_files(path, segment)
        self.root = params.KINETICS_DIR
        self.frames_dir = os.path.join(self.root, '{0}_frames'.format(usage))
        self.video_filenames = [dirname for dirname in os.listdir(self.frames_dir) if os.path.isdir(os.path.join(self.frames_dir, dirname))]
        self.filenames_dict = dict()
        if os.path.exists('./filenames_dict.pkl'):
            with open('./filenames_dict.pkl', 'rb') as handle:
                self.filenames_dict = pickle.load(handle)
        else:
            self.init_frames_count()
            with open('./filenames_dict.pkl', 'wb') as handle:
                pickle.dump(self.filenames_dict, handle)
        self.transform = transforms.Compose([transforms.CenterCrop(720),
                                             transforms.Resize((224,224)),
                                             transforms.ToTensor()])
        
    def init_frames_count(self):
        ind = 0
        n_videos = len(self.video_filenames)
        for v_id, video_fname in enumerate(self.video_filenames):
            video_path = os.path.join(self.frames_dir, video_fname)
            # video_frames_paths = [os.path.join(video_path, fname) for fname in os.listdir(video_path) if fname.endswith('.mp4')]
            n_frames = len(os.listdir(video_path))
            for i in range(0, n_frames-self.window+1, self.step_size):
                self.filenames_dict[ind] = (video_fname, (i, i+self.window-1))
                ind += 1
            if v_id % 1000 == 0:
                print(v_id)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filenames_dict)

    def __getitem__(self, index: int):
        frames = []           
        video_name, (f_start, f_end) = self.filenames_dict[index]
        frames_path = os.path.join(self.frames_dir, video_name)
        for i in range(f_start, f_end+1):
            image_path = os.path.join(frames_path, '{0}.jpg'.format(i))
            frame = self.transform(Image.open(image_path))
            frames.append(frame)
        x = torch.stack(frames, dim=1)

        return x