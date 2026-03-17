import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int=38, freeze_backbone: bool=True) -> nn.Module:

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False  
        
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        return model
    

def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f'Device: {device}')
    return device


if __name__ == '__main__':
    model = build_model(num_classes=18, freeze_backbone=True)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total parameters: {total:,}')
    print(f'Trainable parameters: {trainable:,}')
    print(f'Frozen parameters: {total - trainable}')
    print(f'\nFinal layer: {model.fc}')