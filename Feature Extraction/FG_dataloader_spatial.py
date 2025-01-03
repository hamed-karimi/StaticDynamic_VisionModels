import gc
from datetime import datetime
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
from torch.utils.data.dataset import T_co

class FG_Dataset(Dataset):

    def __init__(self, path, segment, window):
        self.path = path
        self.window = window
        self.segment = segment
        self.transform = transforms.Compose(
            [transforms.CenterCrop(720), 
             transforms.Resize((224, 224)), 
             transforms.ToTensor(),
             # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
             ])
        self.all_frames = self.load_all_files(path, segment)

    def load_all_files(self, root_path, segment):
        folder_name = 'seg_{0}'.format(segment)
        segment_dir = os.path.join(root_path, folder_name)
        frame_num = len(
            [entry for entry in os.listdir(segment_dir) if os.path.isfile(os.path.join(segment_dir, entry))])
        frame_indices = np.expand_dims(np.char.zfill(np.array(range(frame_num), dtype=str), 6), 1)
        frames_path = np.array([segment_dir + '/' + 'f_' + s[0] + '.jpg' for s in frame_indices])
        # first_frame_shape= cv2.imread(frames_path[0]).shape
        pil_image = self.transform(Image.open(frames_path[0]))
        first_frame = np.array(pil_image)
        all_frames = torch.zeros((frame_num, first_frame.shape[0], first_frame.shape[1], first_frame.shape[2]),
                                 dtype=torch.float32)
        now = datetime.now()
        for i in range(frame_num):
            frame = Image.open(frames_path[i])
            if self.transform:
                frame = self.transform(frame)
                
            all_frames[i, :, :, :] = frame
            if (i+1) % 1000 == 0:
                print(i+1, end=' ')
                # break
                
        del first_frame
        del frame_indices
        gc.collect()
        print(datetime.now() - now)
        print('All frames loaded')
        return all_frames

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.all_frames) - self.window + 1

    def __getitem__(self, index: int):
        frames = []
        for i in range(index, index+self.window):
            frames.append(self.all_frames[i, :, :, :])
        x = torch.stack(frames, dim=1).squeeze()

        return x
