import torch
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B,16,64,64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # [B,16,32,32]
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # [B,32,32,32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # [B,32,16,16]
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # [B,64,16,16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)          # [B,64,8,8]
        )
        # Replace the flatten+linear block with Global Average Pooling for efficiency.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # [B,64,1,1]
        self.classifier = nn.Sequential(
            nn.Flatten(),           # [B,64]
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
