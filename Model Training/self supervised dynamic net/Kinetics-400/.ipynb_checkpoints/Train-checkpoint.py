import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import json
from torchvision.datasets import Kinetics
import torchvision.transforms as transforms
import sys
import math
import os
from os.path import join as pjoin
from os.path import exists as pexists
import torchvision.transforms as transforms
from Kinetics_Dataset import KDataset
if '/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/models/' not in sys.path:
    sys.path.append('/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/models/')
if '/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/' not in sys.path:
    sys.path.append('/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/two-stream-pytorch-master/')
if '/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/MotionNet' not in sys.path:
    sys.path.append('/data/karimike/PycharmProjects/pythonProject/Two_Streams_Project/MotionNet')
import feafa_architecture
import feafa_criterion
import feafa_utils
    
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

class Trainer:
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.encoder = encoder.to(self.gpu_id)
        self.encoder.train()
        self.criterion = feafa_criterion.SimpleLoss(self.encoder)
        self.decoder = decoder
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.encoder = DDP(self.encoder, device_ids=[self.gpu_id])
        

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.encoder.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, frames):
        self.optimizer.zero_grad()
        # output = self.model(source)
        # loss = F.cross_entropy(output, targets)
        # loss.backward()
        # self.optimizer.step()
        flows = self.encoder(frames)
        t0s, reconstructed, flows_reshaped = self.decoder(frames, flows) # t0s are original images excluding 
                                                                               # the 11th, downsampled to match the 
                                                                               # reconstructed versions
   
        loss = self.criterion(t0s, reconstructed, flows_reshaped, self.encoder)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        next_data = next(iter(self.train_data))
        print(next_data.shape)
        b_sz = len(next_data[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        
        for frames in self.train_data:
            frames = frames.to(self.gpu_id)
            self._run_batch(frames)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.encoder.state_dict(), # check this
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            if epoch == 0:
                print('Test snapshot saving...')
                self._save_snapshot(epoch)
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def get_parameters():
    with open('./Parameters.json'.format(), 'r') as json_file:
        params = json.load(json_file,
                           object_hook=lambda d: SimpleNamespace(**d))
    return params

def load_train_objs(params):
    train_set = KDataset(usage='train', window=11, step_size=11)
    encoder = feafa_architecture.TinyMotionNet()
    # encoder.train()
    decoder = feafa_utils.Reconstructor()
    # criterion = feafa_criterion.SimpleLoss(encoder)
    optimizer = torch.optim.SGD(encoder.parameters(), 
                                lr=params.LEARNING_RATE,
                                momentum=params.MOMENTUM,
                                weight_decay=params.WEIGHT_DECAY)
    
    return train_set, encoder, decoder, optimizer


def prepare_dataloader(dataset: KDataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(params, save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    dataset, encoder, decoder, optimizer = load_train_objs(params)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(encoder=encoder, 
                      decoder=decoder, 
                      train_data=train_data, 
                      optimizer=optimizer, 
                      save_every=save_every, 
                      snapshot_path=snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    params = get_parameters()
    # parser = argparse.ArgumentParser(description='simple distributed training job')
    # parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    # args = parser.parse_args()

    main(params=params,
         save_every=1,
         total_epochs=params.EPOCH_NUM,
         batch_size=params.BATCH_SIZE)