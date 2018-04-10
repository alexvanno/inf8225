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

import numpy as np
import random


root = './data'
download = True

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

train_set = ds.MNIST(root=root, train=True, transform=trans, download=download)
valid_set = ds.MNIST(root=root, train=True, transform=trans, download=download)
test_set = ds.MNIST(root=root, train=False, transform=trans)

# From https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

batch_size = 50
valid_batch_size = 50
test_batch_size = 50

le_ra = 0.0001
nombre_epochs = 100

train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_set,
    batch_size=valid_batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=test_batch_size, shuffle=True)

# Utility function to transform an image
def to_img(x):
    x = 0.5 * (x+1)
    x = x.clamp(0,1)
    x = x.view(x.size(0),1,28,28)
    return x

class FcNetwork(nn.Module):
    ch_dconv1 = 25
    ch_dconv2 = 50
    #ch_dconv3 = 30
    #ch_uconv1 = 20
    ch_uconv2 = 25
    ch_uconv3 = 1

    hidden_layer_size = 1000
    def __init__(self):
        super().__init__()
        self.dconv1 = nn.Conv2d(1, self.ch_dconv1, kernel_size=3, padding=1, stride=2)
        self.dconv2 = nn.Conv2d(self.ch_dconv1, self.ch_dconv2, kernel_size=3, padding=1, stride=2)
        #self.dconv3= nn.Conv2d(self.ch_dconv2, self.ch_dconv3, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d()
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        self.uconv1 = nn.ConvTranspose2d(self.ch_dconv2, self.ch_uconv2, kernel_size=3, padding=1)
        self.uconv2 = nn.ConvTranspose2d(self.ch_uconv2, self.ch_uconv3, kernel_size=3, padding=1)
        #self.uconv3 = nn.ConvTranspose2d(self.ch_uconv2, self.ch_uconv3, kernel_size=3, padding=1)

    def forward(self, image):
        # Encodeur
        x = F.relu(self.dconv1(image))
        x = F.relu(self.dconv2(self.drop(x)))

        # Decodeur
        x = self.upsamp(x)
        x = F.relu(self.uconv1(x))
        x = self.upsamp(x)
        x = F.tanh(self.uconv2(x))

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

def valid(model, valid_loader) :
    model.eval()
    valid_loss = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True).cuda() , Variable(target).cuda()
        output = model(data)
        valid_loss += nn.MSELoss()(output, data)
    valid_loss /= len(valid_loader.dataset)
    return valid_loss.data[0] * 100000

def test(model, test_loader, noise):
    model.eval()
    test_loss = 0
    for test_data, _ in test_loader:
        test_data = Variable(test_data, volatile=True).cuda()
        pic = to_img(test_data.cpu().data)
        save_image(pic, './dc_img/test_imageOriginalInput.png')
        if noise :
            test_data = test_data + Variable(torch.randn(test_data.size()).cuda() * 1)
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

    for epoch in range(1, epochs + 1):
        model, epoch_loss = train(model, train_loader, optimizer, epoch)
        losses.append(epoch_loss)
        valid_loss = valid(model, valid_loader)
        print("epoch [{}/{}], train loss = {} and valid loss = {}".format(epoch, nombre_epochs, epoch_loss, valid_loss))
        v_losses.append(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model
    r = np.arange(0, epochs)
    plt.plot(r, losses, r, v_losses)
    plt.title("Loss reseau autoencod conv 2 couches\n{} epochs et LR = {}".format(epochs, lr))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    return best_model, best_loss

# Main
best_loss = 0
best_model = []
for model in [FcNetwork()] :
    model.cuda()
    best_model, best_loss = experiment(model)
    # if precision > best_precision:
    #     best_precision = precision
    #     best_model = model
test_loss = test(model, test_loader, noise=True)
