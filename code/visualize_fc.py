import torch
import numpy as np
import sys
sys.path.append("cNets")
from cNet import *
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import glob
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

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


class CNNShow():
	def __init__(self,model):
		self.model = model
		self.model.eval()
		
	def show(self,input_data,device):
		x = input_data
		x = x.to(device)
		self.model = self.model.to(device)
		for layer in self.model.children(): # NOTE: modules() no. _modules.items() ok
			#print(x.shape)
			#print(layer)
			if isinstance(layer,nn.Linear):
				x = x.view(x.size(0),-1)
				return x
			x = layer(x)
		print("no fc!")
		return x



model_dir="../result/taskC/model.pkl"

data_dir = "../data/train/[0-9]" #10 class

model = torch.load(model_dir)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = 224 # set in buildModel
data_transforms = transforms.Compose([
  transforms.Resize(input_size),
  transforms.CenterCrop(input_size),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_datasets = TestDataset(data_dir, data_transforms)
dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=8, shuffle=False, num_workers=4)


CNN = CNNShow(model)
print("CNN show begin")
print()

dir = "../result/visual_res"
if not os.path.exists(dir):
	os.makedirs(dir)

res = None
classes = np.array([])
for inputs,paths in dataloaders:
	for st in paths:
		pos1 = st.rfind("/")
		st = st[:pos1]
		pos2 = st.rfind("/")
		c = st[pos2+1:]
		classes = np.append(classes,int(c))
	out = CNN.show(inputs,device)
	if res is None:
		res = out.detach().cpu().numpy()
	else:
		res = np.vstack((res,out.detach().cpu().numpy()))
	#print(res.shape)
print(res.shape)
print(classes)

# t-SNE
from sklearn.manifold import TSNE
res_embedded = TSNE(n_components=2).fit_transform(res)
print(res_embedded.shape)


# separate
res_dict = {}
for i in range(classes.shape[0]):
	c = classes[i]
	if c not in res_dict.keys():
		v = []
		v.append(res_embedded[i,:])
		res_dict[c]=v
	else:
		res_dict[c].append(res_embedded[i,:])

#print(np.array(res_dict[0]).shape)

plt.figure()
axes = plt.subplot(111)	
colors = cm.rainbow(np.linspace(0, 1, 10))

cs=[]
for i in range(10):
	cs.append(axes.scatter(np.array(res_dict[i])[:,0],np.array(res_dict[i])[:,1], c=colors[i]))
axes.set_title("model C")
axes.legend(iter(cs),res_dict.keys(),loc='best')

#plt.figure()
#for x,c in zip(res_embedded, classes):
#	print(c)
#	plt.scatter(x[0],x[1],c=colors[int(c)])
#	print(colors[int(c)])

plt.savefig(os.path.join(dir,"visual_fc_modelC.png"))

