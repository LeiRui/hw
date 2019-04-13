from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from loadData import *

# Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
data_dir = "/workspace/ruilei/hw/data"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 65

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
feature_extract = True

input_size = 224

# load data
dataloaders_dict = getDataLoader(data_dir, input_size, batch_size)

'''
print('train')
for batch_data,batch_label in dataloaders_dict['train']:
	print(batch_data.size(),batch_label)
print('val')
for batch_data,batch_label in dataloaders_dict['val']:
        print(batch_data.size(),batch_label)
'''

