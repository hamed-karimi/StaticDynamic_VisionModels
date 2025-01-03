import lightning.pytorch as pl
import torch.nn as nn
import torch
import sys
import gc
import os
from datetime import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiplicativeLR
import torchvision.transforms as transforms
# import torch.utils.model_zoo as model_zoo
if '/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/models/' not in sys.path:
    sys.path.append('/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/models/')
if '/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/' not in sys.path:
    sys.path.append('/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/')
if '/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/MotionNet' not in sys.path:
    sys.path.append('/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/MotionNet')
import models
import feafa_architecture
import psutil

    
class FlowOnlyClassifier(nn.Module):
    def __init__(self, model_id,
                 flow_generator,
                 flow_classifier,
                 freeze_flow_generator: bool = True):
        super().__init__()
        assert (isinstance(flow_generator, nn.Module) and isinstance(flow_classifier, nn.Module))

        checkpoint = torch.load('/data/karimike/Documents/forrest_study_fmri/Analysis/All Runs/All models/Model Versions/TinyMotionNet/{0}/logs/version_0/checkpoints/epoch=11-step=148560.ckpt'.format(model_id))
        models_weights_keys = [key for key in checkpoint['state_dict'].keys() if key.startswith('encoder')]
        models_weights = dict()
        for key in models_weights_keys:
            models_weights[key[8:]] = checkpoint['state_dict'][key]
        flow_generator.load_state_dict(models_weights)
        
        if freeze_flow_generator:
            for param in flow_generator.parameters():
                param.requires_grad = False
        self.flow_classifier = flow_classifier
        self.flow_generator = flow_generator
        self.flow_generator.eval()


class LitModel(pl.LightningModule):
    def __init__(self, model_id, lr, momentum, weight_decay, lr_steps=[100, 200]):
        super().__init__()
        self.model = FlowOnlyClassifier(model_id,
                                        feafa_architecture.TinyMotionNet(),
                                        models.flow_resnet.flow_resnet18(pretrained=True, num_classes=500))
        
        self.criterion = nn.CrossEntropyLoss()
        self.all_labels = self.get_all_labels('/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/datasets/HAA500_frames/')
        
        self.register_buffer('lr', lr)
        self.register_buffer('weight_decay', weight_decay)
        self.register_buffer('momentum', momentum)
        self.register_buffer('lr_steps', torch.tensor(lr_steps))

    def get_all_labels(self, frames_dir):
        all_frames = [frame[0:-4] for frame in os.listdir(frames_dir) if frame[-1].isdigit()]
        unique_frame_names = set(all_frames)
        label_indices = {}
        for i, label in enumerate(unique_frame_names):
            label_indices[label] = i
        return label_indices

    def get_batch_label(self, batch_keys):
        # return torch.zeros((len(batch_keys, )))
        batch_labels = torch.Tensor([self.all_labels[k.split('/')[0][:-4]] for k in batch_keys])
        return batch_labels
    
    def training_step(self, batch, batch_idx):
        __key__, frames = batch
        label = self.get_batch_label(__key__).type_as(frames).long()
        if batch_idx % 500 == 0:
            print('cpu memory usage 1: ', psutil.virtual_memory().used / 1000000000)
            # gc.collect()
        with torch.no_grad():
            flows = self.model.flow_generator(frames)
        # flows will return a tuple of flows at different resolutions. Even in eval mode. 0th flow should be
        # at original image resolution (or 1/2)
        y_hat = self.model.flow_classifier(flows[0])
        loss = self.criterion(y_hat, label)
        return loss

    def validation_step(self, batch, batch_idx):
        __key__, frames = batch
        label = self.get_batch_label(__key__).type_as(frames).long()
        with torch.no_grad():
            flows = self.model.flow_generator(frames)
            y_hat = self.model.flow_classifier(flows[0])
        
            top3_labels = y_hat.topk(k=3, dim=1).indices

            loss = self.criterion(y_hat, label)
            pred_acc = torch.zeros_like(top3_labels)
            for d in range(top3_labels.shape[1]):
                pred_acc[:, d] = (label == top3_labels[:, d])

            top1_acc = pred_acc[:, 0]
            top3_acc = pred_acc.sum(dim=1)
            self.log('val_loss', loss, sync_dist=True)
            self.log('val_top1_acc', top1_acc.float().mean(), sync_dist=True)
            self.log('val_top3_acc', top3_acc.float().mean(), sync_dist=True)

    def validation_step_end(self, batch_parts):
        print('validation parts: ', batch_parts.shape)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.flow_classifier.parameters(), 
                                    self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "interval": "epoch",
                "scheduler": MultiplicativeLR(optimizer, 
                                              lr_lambda=lambda x: 0.1 ** (sum(self.current_epoch >= self.lr_steps))),
                "frequency": 1
            },
        }