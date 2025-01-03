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
from transformers import VideoMAEImageProcessor, AutoImageProcessor

class FG_Dataset(Dataset):

    def __init__(self, path, segment, window):
        self.path = path
        self.window = window
        self.segment = segment
        self.transform = transforms.Compose(
            [transforms.CenterCrop(720), 
             transforms.Resize((224, 224)), 
             # transforms.ToTensor()
            ])
        self.all_frames = self.load_all_files(path, segment)
        # self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        # self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.processor = AutoImageProcessor.from_pretrained('MCG-NJU/videomae-base')
        
    def load_all_files(self, root_path, segment):
        folder_name = 'seg_{0}'.format(segment)
        segment_dir = os.path.join(root_path, folder_name)
        frame_num = len(
            [entry for entry in os.listdir(segment_dir) if os.path.isfile(os.path.join(segment_dir, entry))])
        frame_indices = np.expand_dims(np.char.zfill(np.array(range(frame_num), dtype=str), 6), 1)
        frames_path = np.array([segment_dir + '/' + 'f_' + s[0] + '.jpg' for s in frame_indices])
        all_frames = []
        now = datetime.now()
        for i in range(frame_num):
            frame = Image.open(frames_path[i])
            if self.transform:
                frame = self.transform(frame)
            frame = np.array(frame)
            np_frame = np.stack([frame[:, :, 0], 
                                 frame[:, :, 1], 
                                 frame[:, :, 2]], axis=0)
            all_frames.append(np_frame)
            if (i+1) % 1000 == 0:
                print(i+1, end=' ')
                # break
                
        del frame_indices
        gc.collect()
        print(datetime.now() - now)
        return all_frames

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.all_frames) - self.window + 1

    def __getitem__(self, index: int):
        # frames = []
        # print('before:', self.all_frames[0].shape)
        # for i in range(index, index+self.window):
            # at = np.stack([self.all_frames[i][:, :, 0], 
            #                self.all_frames[i][:, :, 1], 
            #                self.all_frames[i][:, :, 2]], axis=0)
            
            # frames.append(at)
            # frames.append(self.all_frames[i, :, :, :])
        # x = torch.stack(frames, dim=1)
        # print('after:', frames[0].shape)
        x = self.processor(self.all_frames[index:index+self.window], return_tensors="pt")['pixel_values']
        return x.squeeze()


# ds = FG_Dataset(r'/data/karimike/Documents/forrest_study_fmri/DNN/Stimulus/Frames/', 1, 11)
# aa = ds.load_all_files(r'/data/karimike/Documents/forrest_study_fmri/DNN/Stimulus/Frames/', 1)
# print(ds.__getitem__(3))
