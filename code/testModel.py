import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
from PIL import Image
import numpy as np
import pandas as pd

#model_dir = "/workspace/ruilei/hw/result/taskA/model.pkl"
model_dir="/workspace/ruilei/hw/result/taskB/model.pkl"
#model_dir="model.pkl"
data_dir = "/workspace/ruilei/hw/data/train/2"

class TestDataset(Dataset):
  def __init__(self, path, transform=None):
    self.image_paths = glob.glob(os.path.join(path,'*.jpg'))
    self.transform = transform

  def __getitem__(self, index):
    path = self.image_paths[index]
    x = Image.open(path)
    if self.transform is not None:
      x = self.transform(x)

    return x,path

  def __len__(self):
    return len(self.image_paths)

print("Loading model...")
model=torch.load(model_dir)
model.eval()
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the model to GPU
model= model.to(device)

print("Initializing Datasets and Dataloaders...")
input_size = 224 # set in buildModel
data_transforms = transforms.Compose([
  transforms.Resize(input_size),
  transforms.CenterCrop(input_size),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_datasets = TestDataset(data_dir, data_transforms)
dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=8, shuffle=False, num_workers=4)

names = np.array([])
res = np.array([])
for inputs,paths in dataloaders:
  for st in paths:
    pos = st.find(".jpg")
    name = st[pos-4:pos]
    names = np.append(names,name)
  
  inputs = inputs.to(device)
  outputs = model(inputs)
  _, preds = torch.max(outputs, 1)
  res = np.append(res,preds.cpu().numpy())

# translate idx to class
idx_to_class = pd.read_csv("idx_to_class.txt")['class']
res_class = []
for i in range(len(res)):
  res_class.append(idx_to_class[res[i]])
res_class = np.array(res_class)

import pandas as pd
# x = pd.DataFrame(columns=['id','pred_idx','pred_class'], data=np.column_stack((names,res,res_class)))
x = pd.DataFrame(columns=['id','pred_class'], data=np.column_stack((names,res_class)))
x.to_csv("test_result.csv",index=False)
print(names)
print(res)
