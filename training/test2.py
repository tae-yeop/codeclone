import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3,4,3)
    def forward(self, x):
        return self.layer(x)


t = torch.randn(3,4)
print(t.requires_grad)
mymodel = MyModel()
mymodel.requires_grad_(False)
print(next(mymodel.layer.named_parameters()))