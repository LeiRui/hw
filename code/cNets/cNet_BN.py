# coding=UTF-8  
import torchvision.models as models  
import torch  
import torch.nn as nn  
import math  
import torch.utils.model_zoo as model_zoo  

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

# Bottleneck of ResNet without batch normalization 
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.conv2 = conv3x3(planes, planes, stride, groups)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
 
class Net(nn.Module):  
  
    def __init__(self, layers=[3,3,3,3], num_classes=65):  
        self.inplanes = 64  
        super(Net, self).__init__()  
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  
                               bias=False)  
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
        #self.maxpool2 = nn.MaxPool2d(32)  
        #去掉原来的fc层，新增一个fclass层  
        self.fclass = nn.Linear(2048, num_classes)  
  
        for m in self.modules():  
            if isinstance(m, nn.Conv2d):  
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  
                m.weight.data.normal_(0, math.sqrt(2. / n))  
  
    def _make_layer(self, block, planes, blocks, stride=1):  
        downsample = None  
        if stride != 1 or self.inplanes != planes * block.expansion:  
            downsample = nn.Sequential(  
                nn.Conv2d(self.inplanes, planes * block.expansion,  
                          kernel_size=1, stride=stride, bias=False),  
            )  
  
        layers = []  
        layers.append(block(self.inplanes, planes, stride, downsample))  
        self.inplanes = planes * block.expansion  
        for i in range(1, blocks):  
            layers.append(block(self.inplanes, planes))  
  
        return nn.Sequential(*layers)  
  
    def forward(self, x):  
        x = self.conv1(x)  
        x = self.relu(x)  
        x = self.maxpool(x)  
  
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x)  
  
        x = self.avgpool(x)  
        x = x.view(x.size(0), -1)  
        x = self.fclass(x)

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
