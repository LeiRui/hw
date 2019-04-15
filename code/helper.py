import numpy as np
import torch
import torch.nn as nn
from loadData import *
from plot_cfm import *
from createOptimizer import *
dir = "/workspace/ruilei/hw/result/taskB"

# plot acc and loss curve
his_dir = "/workspace/ruilei/hw/result/taskB/history.csv"
import pandas as pd
his = pd.read_csv(his_dir)
train_acc_history = his['train_acc_history']
train_loss_history = his['train_loss_history']
val_acc_history=his['val_acc_history']
val_loss_history=his['val_loss_history']

num_epochs = 60
plt.figure()
plt.title("Task B: Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.plot(range(1,num_epochs+1),train_acc_history,label="train")
plt.plot(range(1,num_epochs+1),val_acc_history,label="validation")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0),fontsize=3)
plt.legend()
plt.savefig(os.path.join(dir,"acc.png"),transparent=False,bbox_inches='tight',dpi=500)

plt.figure()
plt.title("Task B: Loss vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Loss")
plt.plot(range(1,num_epochs+1),train_loss_history,label="train")
plt.plot(range(1,num_epochs+1),val_loss_history,label="validation")
#plt.ylim((0,5.))
plt.xticks(np.arange(1, num_epochs+1, 1.0),fontsize=3)
plt.legend()
plt.savefig(os.path.join(dir,"loss.png"),transparent=False,bbox_inches='tight',dpi=500)


# load saved model
model_dir = "/workspace/ruilei/hw/result/taskB/model.pkl"
model = torch.load(model_dir)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# load data
data_dir = "/workspace/ruilei/hw/data"
input_size=224
batch_size=8
dataloaders_dict = getDataLoader(data_dir, input_size, batch_size)


# get best_cfm
best_cfm = np.array([[0 for col in range(65)] for row in range(65)])
model.eval()
for inputs, labels in dataloaders_dict['val']:
	inputs = inputs.to(device)
	labels = labels.to(device)
	with torch.set_grad_enabled(False):
		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)
		for i in range(len(preds)):
			best_cfm[labels.cpu().numpy()[i]][preds.cpu().numpy()[i]] += 1 # results of the current batch

from plot_cfm import *
plot_cfm(best_cfm, dataloaders_dict['train'].dataset.classes, dir)

