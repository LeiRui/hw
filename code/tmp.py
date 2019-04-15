import torch
from cNet import *
model = Net([3,4,3,4])
# model = Net()
input = torch.randn(8,3,224,224)
output = model(input)

