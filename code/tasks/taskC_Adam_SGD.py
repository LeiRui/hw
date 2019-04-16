from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
sys.path.append("../optimizers")
sys.path.append("../cNets")
from buildModel import *
from loadData import *
from createOptimizer_Adam_SGD import *
from trainModel import *
from cNet import *

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
data_dir = "../../data"

# Number of classes in the dataset
num_classes = 65

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 180

# Initialize the model for this run
model_ft = Net()
# Print the model we just instantiated
print(model_ft)

# load data
dataloaders_dict = getDataLoader(data_dir, 224, batch_size)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# create optimizer
optimizer_ft = createOptimizer(model_ft, feature_extract=False)

# Detect if we have a GPU available
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# Train and evaluate
model_ft, train_loss_history, train_acc_history, val_loss_history, \
val_acc_history, best_cfm \
  = train_model("taskC_Adam_SGD", device, model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=False)

import time
import os
now = time.strftime("%Y-%m-%d-%H_%M",time.localtime(time.time()))
dir = "../../result/taskC_Adam_SGD_"+now
if not os.path.exists(dir):
  print("creating directory: ", dir)
  os.makedirs(dir)

torch.save(model_ft, os.path.join(dir,"model.pkl"))
# torch.save(model_object.state_dict(), '/workspace/ruilei/hw/result/params.pkl')
name = ['train_loss_history','train_acc_history','val_loss_history','val_acc_history']
history = pd.DataFrame(columns=name, data=np.array([train_loss_history, train_acc_history, val_loss_history,val_acc_history]).T)
history.to_csv(os.path.join(dir,"history.csv"),index_label="epoch")
# np.savetxt(os.path.join(dir,"cfm.txt"), best_cfm)

#lidation Accuracy vs. Number of Training Epochs")
plt.figure()
plt.title("Task C: Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.plot(range(1,num_epochs+1),train_acc_history,label="train")
plt.plot(range(1,num_epochs+1),val_acc_history,label="validation")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0),fontsize=3)
plt.legend()
plt.savefig(os.path.join(dir,"acc.png"),transparent=False,bbox_inches='tight',dpi=500)


plt.figure()
plt.title("Task C: Loss vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Loss")
plt.plot(range(1,num_epochs+1),train_loss_history,label="train")
plt.plot(range(1,num_epochs+1),val_loss_history,label="validation")
#plt.ylim((0,5.))
plt.xticks(np.arange(1, num_epochs+1, 1.0),fontsize=3)
plt.legend()
plt.savefig(os.path.join(dir,"loss.png"),transparent=False,bbox_inches='tight',dpi=500)

from plot_cfm import *
plot_cfm(best_cfm, dataloaders_dict['train'].dataset.classes, dir)

