# coding=UTF-8  
import torchvision.models as models  
import torch  
import torch.nn as nn  
import math  
import torch.utils.model_zoo as model_zoo  
from torchvision.models.resnet import Bottleneck, ResNet
  
class Net(nn.Module):  
  
    def __init__(self, layers=[3,3,3,3], num_classes=65):  
        self.inplanes = 64  
        super(Net, self).__init__()  
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  
                               bias=False)  
        self.bn1 = nn.BatchNorm2d(64)  
        self.relu = nn.ReLU(inplace=True)  
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])  
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)  
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)  
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)  
        self.avgpool = nn.AvgPool2d(7, stride=1)  
        #新增一个反卷积层  
        #self.convtranspose1 = nn.ConvTranspose2d(2048, 2048, kernel_size=14, stride=3)
        #新增一个最大池化层  
        #iself.maxpool2 = nn.MaxPool2d(32)  
        self.dropout = nn.Dropout(p=0.5)
        #去掉原来的fc层，新增一个fclass层  
        self.fclass = nn.Linear(2048, num_classes)  
  
        for m in self.modules():  
            if isinstance(m, nn.Conv2d):  
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  
                m.weight.data.normal_(0, math.sqrt(2. / n))  
            elif isinstance(m, nn.BatchNorm2d):  
                m.weight.data.fill_(1)  
                m.bias.data.zero_()  
  
    def _make_layer(self, block, planes, blocks, stride=1):  
        downsample = None  
        if stride != 1 or self.inplanes != planes * block.expansion:  
            downsample = nn.Sequential(  
                nn.Conv2d(self.inplanes, planes * block.expansion,  
                          kernel_size=1, stride=stride, bias=False),  
                nn.BatchNorm2d(planes * block.expansion),  
            )  
  
        layers = []  
        layers.append(block(self.inplanes, planes, stride, downsample))  
        self.inplanes = planes * block.expansion  
        for i in range(1, blocks):  
            layers.append(block(self.inplanes, planes))  
  
        return nn.Sequential(*layers)  
  
    def forward(self, x):  
        x = self.conv1(x)  
        # print("conv1",x.shape)
        x = self.bn1(x)  
        # print("bn1",x.shape)
        x = self.relu(x)  
        # print("relu",x.shape)
        x = self.maxpool(x)  
        # print("maxpool",x.shape)
  
        x = self.layer1(x)  
        # print("layer1",x.shape)
        x = self.layer2(x)  
        # print("layer2",x.shape)
        x = self.layer3(x)  
        # print("layer3",x.shape)
        x = self.layer4(x)  
        # print("layer4",x.shape)
  
        x = self.avgpool(x)  
        #print("avgpool",x.shape)
        #新加层的forward  
        x = x.view(x.size(0), -1)  
        #print("view",x.shape)
        #x = self.convtranspose1(x)  
        # print("convtranspose",x.shape)
        # x = self.maxpool2(x)  
        # print("maxpool",x.shape)
        # x = x.view(x.size(0), -1)  
        # print("view",x.shape)
        x = self.dropout(x)
        x = self.fclass(x)  
        # print("fc",x.shape)
  
        return x  
  
#加载model  
# resnet50 = models.resnet50(pretrained=True)  
# cnn = Net([3, 4, 6, 3])  
#读取参数  
# pretrained_dict = resnet50.state_dict()  
# model_dict = cnn.state_dict()  
# 将pretrained_dict里不属于model_dict的键剔除掉  
# pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}  
# 更新现有的model_dict  
# model_dict.update(pretrained_dict)  
# 加载我们真正需要的state_dict  
# cnn.load_state_dict(model_dict)  
# print(resnet50)  
# print(cnn)  
