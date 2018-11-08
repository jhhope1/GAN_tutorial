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
import os.path
epoch=100
mini_batch=100
cuda = torch.device('cuda:0')
max_accuracy=0

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
train_loader = torch.utils.data.DataLoader(datasets.MNIST('\\data', train=True, download=True,transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),batch_size=mini_batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('\\data', train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor()])),batch_size=len(mnist_testset), shuffle=False)
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
    plt.imshow(npimg.astype(np.uint8),cmap=plt.cm.binary)
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def make_mini_batch_cri(data_,target_):
    data_.requires_grad=False
    target_.requires_grad=False


    id,dis=Gen_input(mini_batch)
    id.requires_grad=False
    dis.requires_grad=False
    gen_input=torch.cat((id,dis),dim=1).cuda()
    
    x=torch.cat((data_,generator_net(gen_input)))
    target=torch.cat((target_,id))
    target_certificate=torch.cat((torch.full([mini_batch],1.),torch.zeros((mini_batch)))).cuda()
    target=torch.cat((target,target_certificate.reshape([-1,1])),dim=1)
    return x,target

def make_mini_batch_gen():
    target,dis=Gen_input(mini_batch)
    target.requires_grad=False
    dis.requires_grad=False
    gen_input=torch.cat((target,dis),dim=1).cuda()
    
    x=generator_net(gen_input)
    target_certificate=torch.full([mini_batch],1.).cuda()
    target=torch.cat((target,target_certificate.reshape([-1,1])),dim=1)


    return x,target



def Gen_input(i):
    torch.randn((i,10))
    rd=np.random.random_integers(0,9,(i))
    id=torch.Tensor(one_hot(rd)).cuda()
    dis=torch.randn(i,10).cuda()
    #gen_input = torch.cat((id,dis)).cuda()
    return id,dis

class CriticNet(nn.Module):#first 10 number judge the lables, last one judge that data has generated
    _loss = torch.nn.BCEWithLogitsLoss()
    def __init__(self):
        super(CriticNet,self).__init__()
        self.full0=nn.Linear(image_size, 500)
        self.full1=nn.Linear(500,1000)
        self.full2=nn.Linear(1000,10)

        self.full0_=nn.Linear(image_size, 500)
        self.full1_=nn.Linear(500,1000)
        self.full3_=nn.Linear(1000,1)

        self.R=nn.LeakyReLU(0.2)
    def forward(self,input):
        input1=self.R(self.full0(input))
        input1=self.R(self.full1(input1))
        input1=self.full2(input1)
        input1=torch.softmax(input1,dim=1)

        input2=self.R(self.full0_(input))
        input2=self.R(self.full1_(input2))
        input2=torch.sigmoid(self.full3_(input2))

        input=torch.cat((input1, input2),dim=1)
        return input

class GeneratorNet(nn.Module):
    _loss = torch.nn.BCEWithLogitsLoss()

    def __init__(self):
        super(GeneratorNet,self).__init__()
        self.full1=nn.Linear(20,30)
        self.full2=nn.Linear(30,40)
        self.full3=nn.Linear(40,image_size)

    def forward(self,input):
        input=F.relu(self.full1(input))
        input=F.relu(self.full2(input))
        input=F.relu(self.full3(input))
        input=torch.tanh(input)
        return input

def pred():
    global max_accuracy
    for ind, (data,target) in enumerate(test_loader):
        result = critic_net(data.cpu().cuda())
        indices = torch.tensor([0,1,2,3,4,5,6,7,8,9]).cuda()
        result = torch.index_select(result, 1, indices)
        error = CriticNet._loss(result,torch.Tensor(one_hot(target)).cuda())
        ans = torch.sum(torch.eq(torch.argmax(result,dim=1),torch.argmax(torch.Tensor(one_hot(target)).cuda(),dim=1)))
        print("total data number = ",data.shape[0],"\nans = ",ans.cpu().numpy(),"\naccuracy = ",ans.cpu().numpy()/data.shape[0]*100,"%\nerror = ",error.cpu().detach().numpy())
        max_accuracy=max(max_accuracy,ans.cpu().numpy()/data.shape[0]*100)
        if(max_accuracy==ans.cpu().numpy()/data.shape[0]*100):
            save()

def predgen():
    x,target=make_mini_batch_gen()
    loss=GeneratorNet._loss(critic_net(x),target)
    print("genloss = ",loss.detach().cpu().numpy())

def save():
    torch.save(critic_net,"cri.pt")
    torch.save(generator_net,"gen.pt")
def genload():
    return torch.load("gen.pt")
def criload():
    return torch.load("cri.pt")

critic_net = CriticNet()
generator_net = GeneratorNet()
if(os.path.isfile("cri.pt")):
    critic_net=criload()
if(os.path.isfile("gen.pt")):
    gen_net=genload()
critic_net.cuda()
generator_net.cuda()


def critic_bpp(data_,target_):
    generator_net.requires_grad=False
    critic_net.requires_grad=True
    optimizer_critic.zero_grad()
    x,target=make_mini_batch_cri(data_,target_)
    loss=CriticNet._loss(critic_net(x),target).cuda()
    loss.backward()
    optimizer_critic.step()

def generator_bpp():
    critic_net.requires_grad=False
    generator_net.requires_grad=True
    optimizer_generator.zero_grad()
    x,target=make_mini_batch_gen()
    loss=GeneratorNet._loss(critic_net(x),target)
    loss.backward()
    optimizer_generator.step()
    


optimizer_critic = torch.optim.SGD(critic_net.parameters(),lr=0.1)
optimizer_generator = torch.optim.SGD(generator_net.parameters(),lr=0.1)

for e in range(epoch):
    print(e)
    pred()
    predgen()
    for ind, (data,target)in enumerate(train_loader):
        data, target=data.cuda(), torch.Tensor(one_hot(target)).cuda()
        critic_bpp(data,target)
        generator_bpp()
    gen_input=Gen_input(1)
    print("number = ",torch.argmax(gen_input[0]).detach().cpu().numpy())
    gen_input=torch.cat(gen_input,dim=1)
    gen_img = generator_net(gen_input)
    imshow(gen_img.reshape((28,28)).detach().cpu().numpy())