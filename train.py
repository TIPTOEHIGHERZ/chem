import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from network.modeling import deeplabv3plus_resnet101
from network._deeplab import DeepLabV3
import tqdm


def frozen_param(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    
    return


def unfrozen_param(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True

    return


def train(model: nn.Module, 
          data_loader: DataLoader, 
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device='cuda'):
    
    data = tqdm.tqdm(data_loader)
    model.train()

    for x, y in data:
        x.to(device)
        y.to(device)

        



num_classes = 21
output_stride = 16
layers = [3, 4, 23, 3]
model: DeepLabV3 = deeplabv3plus_resnet101(num_classes=num_classes,
                                output_stride=output_stride,
                                pretrained_backbone=True)
frozen_param(model.backbone)
