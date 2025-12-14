import torch.nn as nn
import torchvision.models as models


class Person_Classifer(nn.Module):
    def __init__(self, num_classes):
        super(Person_Classifer, self).__init__()
        
        self.resnet50 = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.resnet50(x)      # (batch, 2048, 1, 1)
        x = x.view(b, -1)         # (batch, 2048)
        x = self.fc(x)            # (batch, num_class)          
        return x
