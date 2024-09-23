import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from network.modeling import deeplabv3plus_resnet101
from network._deeplab import DeepLabV3
from typing import Callable
from metrics import StreamSegMetrics
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
          metric: StreamSegMetrics,
          loss_fn: Callable = nn.CrossEntropyLoss(ignore_index=255, reduction='mean'),
          lr_scheduler=None,
          device='cuda'):
    
    data = tqdm.tqdm(data_loader)
    model.to(device)
    model.train()

    metric.reset()
    for epoch in range(epochs):
        for x, y in data:
            optimizer.zero_grad()
            x.to(device)
            y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            loss.backward()
            optimizer.step()

            metric.update(y.cpu().numpy(), y_pred.cpu().numpy())
            score = metric.get_results()

            mean_iou = score['Mean IoU']
            class_iou = score['Class IoU']
            acc = score['Overall Acc']
            data.set_description(f'{epoch} / {epochs}: ')
            data.set_postfix({'acc': acc, 'mean iou': mean_iou, 'class_iou': class_iou})

        if lr_scheduler:
            lr_scheduler.step()

    return


def main():
    num_classes = 21
    output_stride = 16
    metric = StreamSegMetrics(num_classes)
    model: DeepLabV3 = deeplabv3plus_resnet101(num_classes=num_classes,
                                               output_stride=output_stride,
                                               pretrained_backbone=True)
    lr = 1e-2
    step_size = 10000
    optimizer = Adam(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * lr},
        {'params': model.classifier.parameters(), 'lr': lr},
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    frozen_param(model.backbone)

    epochs = 100
    train(model, None, optimizer, epochs, metric, nn.CrossEntropyLoss(ignore_index=255, reduction='mean'),
          lr_scheduler=scheduler, device='cuda')


if __name__ == '__main__':
    main()
