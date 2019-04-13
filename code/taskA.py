from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
num_epochs = 2

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
model_ft, train_loss_history, train_acc_history, val_loss_history, \
val_acc_history, best_cfm \
  = train_model(device, model_ft, dataloaders_dict, criterion, optimizer_ft,
                num_epochs=num_epochs, is_inception=(model_name=="inception"))

import time
import os
now = time.strftime("%Y-%m-%d-%H_%M",time.localtime(time.time()))
dir = "/workspace/ruilei/hw/result/taskA_"+now
if not os.path.exists(dir):
  print("creating directory: ", dir)
  os.makedirs(dir)

torch.save(model_ft, os.path.join(dir,"model.pkl"))
# torch.save(model_object.state_dict(), '/workspace/ruilei/hw/result/params.pkl')
name = ['train_loss_history','train_acc_history','val_loss_history','val_acc_history']
history = pd.DataFrame(columns=name, data=np.array([train_loss_history, train_acc_history, val_loss_history,val_acc_history]).T)
history.to_csv(os.path.join(dir,"history.csv"))
np.savetxt(os.path.join(dir,"cfm.txt"), best_cfm)

