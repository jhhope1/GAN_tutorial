import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pathlib
import time

class CriticNet(nn.Module):#first 10 number judge the lables, last one judge that data has generated
    _loss = torch.nn.BCELoss()
    def __init__(self):
        super(CriticNet,self).__init__()
        self.full1_=nn.Linear(784,500)
        self.full2_=nn.Linear(500,100)
        self.full3_=nn.Linear(100,1)
    def forward(self,input):
        input=input.reshape(-1,784)
        input2=F.relu(self.full1_(input))
        input2=F.relu(self.full2_(input2))
        input2=torch.sigmoid(self.full3_(input2))
        return input2

class GeneratorNet(nn.Module):
    _loss = torch.nn.BCELoss()

    def __init__(self):
        super(GeneratorNet,self).__init__()
        self.full1=nn.Linear(10,100)
        self.full2=nn.Linear(100,500)
        self.full3=nn.Linear(500,28*28)
        torch.nn.init.normal_(self.full1.weight, mean=0,std=0.1)
        torch.nn.init.normal_(self.full2.weight, mean=0,std=0.1)
        torch.nn.init.normal_(self.full3.weight, mean=0,std=0.1)
        #torch.nn.init.normal_(self.conv1.weight, mean=0,std=1)
        #torch.nn.init.normal_(self.conv2.weight, mean=0,std=1)
        #torch.nn.init.normal_(self.conv3.weight, mean=0,std=1)


    def forward(self,input):
        input=F.relu(self.full1(input))
        input=F.relu(self.full2(input))
        input=F.relu(self.full3(input))
        input=torch.sigmoid(input)
        input=input.reshape((-1,1,28,28))
        return input    
generator_net=GeneratorNet()
critic_net=CriticNet()
generator_net.cuda()
critic_net.cuda()
generator_net.parameters=torch.load("gen.pkl")
critic_net.parameters=torch.load("cri.pkl")
def one_hot(x):
    y=np.zeros((len(x),10))
    for i in range(len(x)):
        y[i][x[i]]=1.
    return y
def imshow(img):
    #img=img/2+0.5
    npimg=img
    if(len(img.shape)==3):
        npimg = npimg.transpose((1,2,0))
    print(img.shape)
    plt.imshow(npimg,cmap=plt.cm.binary)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def Gen_input(i):
    torch.randn((i,10))
    rd=np.random.random_integers(0,9,(i))
    id=torch.Tensor(one_hot(rd)).cuda()
    dis=torch.randn(i,10).cuda()
    #gen_input = torch.cat((id,dis)).cuda()
    return id,dis

gen_input=Gen_input(1)
gen_input=gen_input[1]
print(gen_input)
gen_img = generator_net(gen_input.cuda())
imshow(gen_img.reshape((28,28)).detach().cpu().numpy())