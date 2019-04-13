from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from buildModel import *
from loadData import *
from createOptimizer import *
from trainModel import *

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

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

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Print the model we just instantiated
print(model_ft)

# load data
dataloaders_dict = getDataLoader(data_dir, input_size, batch_size)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# create optimizer
optimizer_ft = createOptimizer(model_ft, feature_extract)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Train and evaluate
model_ft, hist = train_model(device, model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


