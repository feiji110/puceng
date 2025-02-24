import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResNetBlock, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        
    def forward(self, x):
        residual = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out =  self.act(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out =  self.act(out)
        
        return out
