import torch
from cNet import *
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import os

class CNNShow():
	def __init__(self,model):
		self.model = model
		self.model.eval()
		
	def show(self,ImagePath,childNo,device):
		image = self.loadImage(ImagePath)
		cnt = 0
		x = image
		x = x.to(device)
		self.model = self.model.to(device)
		for layer in self.model.children(): # NOTE: modules() no. _modules.items() ok
			#print(x.shape)
			#print(layer)
			if isinstance(layer,nn.Linear):
				x = x.view(x.size(0),-1)
			x = layer(x)
			#if isinstance(layer,nn.Conv2d):
			#	print("!!!!this is a conv2d layer")
			#	cnt += 1
			cnt += 1
			print("conv:", cnt)
			if cnt == childNo:
				print(x.shape)
				return x
		print("childNo is out of range!")
		return

	def loadImage(self,ImagePath):
		image = Image.open(ImagePath)
		data_transforms = transforms.Compose([
			transforms.Resize(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		imData = data_transforms(image)
		imData = Variable(torch.unsqueeze(imData, dim=0), requires_grad=True) #batch (1,3,224,224)
		#print(imData.shape)
		return imData

model = torch.load("/workspace/ruilei/hw/result/taskC/model.pkl")
path = "/workspace/ruilei/hw/data/train/0/0000.jpg"
CNN = CNNShow(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CNN show begin")
print()

dir = "../result/visual_res"
if not os.path.exists(dir):
	os.makedirs(dir)

obj = [1,5,6,7,8]
for n in range(5):
	out = CNN.show(path,obj[n],device)
	plt.figure()
	for i in range(16):
		ax = plt.subplot(4, 4, i + 1)
		ax.set_title('Filter #{}'.format(i), fontsize = 3)
		ax.axis('off')
		plt.imshow(out.data.cpu().numpy()[0,i,:,:],cmap='jet')
		#plt.show()
	plt.savefig(os.path.join(dir,"visual_out_"+str(obj[n])+".png"),transparent=False,bbox_inches='tight',dpi=800)


