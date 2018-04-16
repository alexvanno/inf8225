import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torchvision.datasets as ds
#from fashion import FashionMNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch import nn
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import random

root = './data'
download = True

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

train_set = ds.MNIST(root=root, train=True, transform=trans, download=download)
valid_set = ds.MNIST(root=root, train=True, transform=trans, download=download)
test_set = ds.MNIST(root=root, train=False, transform=trans)

# From https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

batch_size = 100
valid_batch_size = 100
test_batch_size = 50

le_ra = 0.000001
nombre_epochs = 200

oneall = (torch.ones(10000)).byte()
sampler = (torch.round(torch.rand(10000))).byte()

num_train = len(train_set)
indices = list(range(num_train))
split = int(np.floor(0.25 * num_train))

np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=batch_size, shuffle=False, sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(valid_set,
    batch_size=valid_batch_size, shuffle=False, sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=test_batch_size, shuffle=True)

# Utility function to transform an image
def to_img(x):
    x = 0.5 * (x+1)
    x = x.clamp(0,1)
    x = x.view(x.size(0),1,28,28)
    return x

class FcNetwork(nn.Module):
    ch_dconv1 = 16
    ch_dconv2 = 32
    ch_uconv2 = 16
    ch_uconv3 = 1

    hidden_layer_size = 1000
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        assert self.arch < 5
        # Arch 1
        self.dconv1 = nn.Conv2d(1, self.ch_dconv1, kernel_size=3, padding=1, stride=2)
        self.dconv2 = nn.Conv2d(self.ch_dconv1, self.ch_dconv2, kernel_size=3, padding=1, stride=2)
        self.drop = nn.Dropout2d()
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        self.uconv1 = nn.ConvTranspose2d(self.ch_dconv2, self.ch_uconv2, kernel_size=3, padding=1)
        self.uconv2 = nn.ConvTranspose2d(self.ch_uconv2, self.ch_uconv3, kernel_size=3, padding=1)

        # Arch 2
        self.d1 = nn.Conv2d(1, 8, kernel_size=3, padding=0, stride=1)
        self.d2 = nn.Conv2d(8, 16, kernel_size=3, padding=0, stride=1)
        self.d3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.d4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)

        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        self.u4 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1)
        self.u3 = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.u2 = nn.ConvTranspose2d(16, 8, kernel_size=3, padding=0)
        self.u1 = nn.ConvTranspose2d(8, 1, kernel_size=3, padding=0)

        # Arch 3 (ajouts dans le milieu de la couche)
        self.d3point5 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.u3point5 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=0)

        # Arch 4
        self.d4point5 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.d5point5 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.u4point5 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=0)
        self.u5point5 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=0)



#A mettre quand on herite __init__ __len__ __getItem__
    def forward(self, image):
        if (self.arch == 1) :
            # Arch 1
            # Encodeur
            x = F.relu(self.dconv1(image))
            x = F.relu(self.dconv2(x))

            # Decodeur
            x = self.upsamp(x)
            x = F.relu(self.uconv1(x))
            x = self.upsamp(x)
            x = F.tanh(self.uconv2(x))

        elif (self.arch == 2) :
            # Arch 2
            # Encodeur
            x = F.relu(self.d1(image))
            x = F.relu(self.d2(x))
            x = F.relu(self.d3(x))
            x = F.relu(self.d4(x))
            # Décodeur
            x = F.relu(self.u4(self.upsamp(x)))
            x = F.relu(self.u3(self.upsamp(x)))
            x = F.relu(self.u2(x))
            x = F.tanh(self.u1(x))

        elif (self.arch == 3) :
            # Arch 3
            # Encodeur
            x = F.relu(self.d1(image))
            x = F.relu(self.d2(x))
            x = F.relu(self.d3(x))
            x = F.relu(self.d3point5(x))
            x = F.relu(self.d4(x))
            # Décodeur
            x = F.relu(self.u4(self.upsamp(x)))
            x = F.relu(self.u3point5(x))
            x = F.relu(self.u3(self.upsamp(x)))
            x = F.relu(self.u2(x))
            x = F.tanh(self.u1(x))

        elif (self.arch == 4) :
            # Arch 4
            # Encodeur
            x = F.relu(self.d1(image))
            x = F.relu(self.d2(x))
            x = F.relu(self.d3(x))
            x = F.relu(self.d3point5(x))
            x = F.relu(self.d4point5(x))

            # Décodeur
            x = F.relu(self.u4point5(x))
            x = F.relu(self.u3point5(x))
            x = F.relu(self.u3(self.upsamp(x)))
            x = F.relu(self.u2(x))
            x = F.tanh(self.u1(x))

        return x

def train(model, train_loader, optimizer, epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(data).cuda()
        optimizer.zero_grad()
        output = model(data)  # calls the forward function
        loss = nn.MSELoss()(output, data)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    epoch_loss /= len(train_loader.dataset)
    pic = to_img(data.cpu().data)
    save_image(pic, './dc_img/NewimageInput_{}.png'.format(epoch))
    outpic = to_img(output.cpu().data)
    save_image(outpic, './dc_img/NewimageOutput_{}.png'.format(epoch))
    return model, epoch_loss.data[0] * 100000

def valid(model, valid_loader, epoch) :
    model.eval()
    valid_loss = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = model(data)
        valid_loss += nn.MSELoss()(output, data)
    valid_loss /= len(valid_loader.dataset)
    #pic = to_img(data.cpu().data)
    #save_image(pic, './dc_img/NewvalidimageInput_{}.png'.format(epoch))
    #outpic = to_img(output.cpu().data)
    #save_image(outpic, './dc_img/NewvalidimageOutput_{}.png'.format(epoch))
    return valid_loss.data[0] * 100000

def test(model, test_loader, noise):
    model.eval()
    test_loss = 0
    for test_data, _ in test_loader:
        test_data = Variable(test_data, volatile=True).cuda()
        pic = to_img(test_data.cpu().data)
        save_image(pic, './dc_img/test_imageOriginalInput.png')
        if noise :
            test_data = test_data + Variable(torch.randn(test_data.size()) * 1).cuda()
        output = model.forward(test_data)
        test_loss += nn.MSELoss()(output, test_data).data[0]
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/test_imageOutput.png')
        pic = to_img(test_data.cpu().data)
        save_image(pic, './dc_img/test_imageInput.png')
    return (test_loss / len(train_loader.dataset))
    #return test_loss

def experiment(model, epochs=nombre_epochs, lr=le_ra):
    best_loss = 1000000
    best_model = []
    losses = []
    v_losses = []

    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model, epoch_loss = train(model, train_loader, optimizer, epoch)
        losses.append(epoch_loss)
        valid_loss = valid(model, valid_loader, epoch)
        print("epoch [{}/{}], train loss = {} and valid loss = {}".format(epoch, nombre_epochs, epoch_loss, valid_loss))
        v_losses.append(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model
    r = np.arange(0, epochs)
    plt.plot(r, losses, r, v_losses)
    plt.title("Loss architecture 3\n{} epochs et LR = {}".format(epochs, lr))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    return best_model, best_loss

# Main
best_loss = 0
best_model = []
for model in [FcNetwork(3)] :
    model.cuda()
    best_model, best_loss = experiment(model)
test_loss = test(model, test_loader, noise=True)
