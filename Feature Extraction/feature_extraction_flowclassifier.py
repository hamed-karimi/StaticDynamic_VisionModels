import torch
from torch import nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import torchlens as tl
from datetime import datetime
import visualpriors
import sys
import gc
import psutil
import os

from torch.utils.data import BatchSampler, SequentialSampler, DataLoader
from FG_dataloader_flowclassifier import FG_Dataset

import flow_resnet
import feafa_architecture

class FlowOnlyClassifier(nn.Module):
    def __init__(self,
                 flow_generator,
                 flow_classifier,
                 freeze_flow_generator: bool = True):
        super().__init__()
        assert (isinstance(flow_generator, nn.Module) and isinstance(flow_classifier, nn.Module))
       
        if freeze_flow_generator:
            for param in flow_generator.parameters():
                param.requires_grad = False
        self.flow_classifier = flow_classifier
        self.flow_generator = flow_generator
        self.flow_generator.eval()
    def forward(self, batch):
        with torch.no_grad():
            flows = self.flow_generator(batch)
        
        return self.flow_classifier(flows[0])
    
def random_initialization(net):
    if isinstance(net, nn.Linear) or isinstance(net, nn.BatchNorm2d):
        torch.nn.init.uniform_(net.weight, a=-1.0, b=1.0)
        torch.nn.init.uniform_(net.bias, a=-1.0, b=1.0)
        
    if isinstance(net, nn.Conv2d):
        torch.nn.init.uniform_(net.weight, a=-1.0, b=1.0)
def create_model(model_name, mode, device):   
    if model_name == 'Spatial':
        if mode == 'ImageNet-trained':
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights, progress=False)
        elif mode == 'HAA-trained':
            weights = torch.load('/data/karimike/Documents/forrest_study_fmri/Analysis/All Runs/Spatial Stream/weights_SpatialStream.tar')
            model = resnet18(progress=False, num_classes=500).eval()

            old_fc = ['fc_action.weight', 'fc_action.bias']
            new_fc = ['fc.weight', 'fc.bias']
            for i in range(len(old_fc)):
                if old_fc[i] in weights['state_dict'].keys():
                    weights['state_dict'][new_fc[i]] = weights['state_dict'].pop(old_fc[i])
            model.load_state_dict(weights['state_dict'])
        elif mode == 'untrained':
            model = resnet18(pretrained=False, progress=False)
            model.apply(random_initialization)
        
    elif model_name == 'FlowClassifier':
        flow_classifier = models.flow_resnet.flow_resnet18(pretrained=False, num_classes=500)
        model = FlowOnlyClassifier(flow_generator=feafa_architecture.TinyMotionNet(),
                                   flow_classifier=flow_classifier)
        if mode == 'HAA-trained':
            weights = torch.load('/data/karimike/Documents/forrest_study_fmri/Analysis/All Runs/Flow Classifier/model_best.pth.tar')
            model.load_state_dict(weights['state_dict'])
        elif mode == 'untrained':
            model.apply(random_initialization)
            
    return model.to(device)

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
        
model_name = 'FlowClassifier'
seg = int(sys.argv[1])
mode = sys.argv[2] #'HAA-trained'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {0}: '.format(device))

window = 11
save_batch = 5000
batch_size = 500 #50
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

model = create_model(model_name=model_name, mode=mode, device=device)
model.eval()

result_dir = get_result_dir('./Models/{0}'.format(model_name), mode, seg)
print(result_dir)

modules_of_interest = ['flow_classifier.conv1', 
                       'flow_classifier.maxpool', 
                       'flow_classifier.layer1.0.conv2', 
                       'flow_classifier.layer1.1.conv2',            
                       'flow_classifier.layer2.0.conv2', 
                       'flow_classifier.layer2.1.conv2', 
                       'flow_classifier.layer3.0.conv2', 
                       'flow_classifier.layer3.1.conv2', 
                       'flow_classifier.layer4.0.conv2', 
                       'flow_classifier.layer4.1.conv2', 
                       'flow_classifier.avgpool']

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