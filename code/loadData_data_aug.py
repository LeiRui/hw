# Data augmentation and normalization for training
# Just normalization for validation

import torch
from torchvision import datasets, transforms
import os
import pandas as pd

def getDataLoader(data_dir, input_size, batch_size):
  data_transforms = {
    'train': transforms.Compose([
      transforms.RandomResizedCrop(input_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
      transforms.Resize(input_size),
      transforms.CenterCrop(input_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }

  print("Initializing Datasets and Dataloaders...")

  # Create training and validation datasets
  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
  # Create training and validation dataloaders
  dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

  # print(image_datasets['train'].class_to_idx)
  if not os.path.exists("idx_to_class.txt"):
    idx_to_class = pd.DataFrame(columns=['class'], data=image_datasets['train'].classes)
    idx_to_class.to_csv("idx_to_class.txt")

  return dataloaders_dict

