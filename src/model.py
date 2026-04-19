import torch
import torch.nn as nn
from torchvision import models

class BrainTumorModel(nn.Module):
    def __init__(self, num_classes: int):
        super(BrainTumorModel, self).__init__()
        self.model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze last block
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace classifier
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)



# if __name__ == "__main__":
#     model = models.resnet50(pretrained=True)
#     for param in model.parameters():
#         param.requires_grad = False 