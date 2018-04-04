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
test_set = ds.MNIST(root=root, train=False, transform=trans)

batch_size = 50
test_batch_size = 50

le_ra = 0.0001
nombre_epochs = 20

train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=test_batch_size, shuffle=True)

losses = []

class FcNetwork(nn.Module):
    ch_dconv1 = 10
    ch_dconv2 = 20
    #ch_dconv3 = 30
    #ch_uconv1 = 20
    ch_uconv2 = 10
    ch_uconv3 = 1

    hidden_layer_size = 1000
    def __init__(self):
        super().__init__()
        self.dconv1 = nn.Conv2d(1, self.ch_dconv1, kernel_size=3, padding=1, stride=2)
        self.dconv2 = nn.Conv2d(self.ch_dconv1, self.ch_dconv2, kernel_size=3, padding=1, stride=2)
        #self.dconv3= nn.Conv2d(self.ch_dconv2, self.ch_dconv3, kernel_size=3, padding=1)
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
        self.uconv1 = nn.ConvTranspose2d(self.ch_dconv2, self.ch_uconv2, kernel_size=3, padding=1)
        self.uconv2 = nn.ConvTranspose2d(self.ch_uconv2, self.ch_uconv3, kernel_size=3, padding=1)
        #self.uconv3 = nn.ConvTranspose2d(self.ch_uconv2, self.ch_uconv3, kernel_size=3, padding=1)

    def forward(self, image):
        # Encodeur
        x = F.relu(self.dconv1(image))
        x = F.relu(self.dconv2(x))

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
    print(epoch_loss.data[0])
    pic = to_img(data.cpu().data)
    save_image(pic, './dc_img/NewimageInput_{}.png'.format(epoch))
    outpic = to_img(output.cpu().data)
    save_image(outpic, './dc_img/NewimageOutput_{}.png'.format(epoch))
    return model

# def valid(model, valid_loader):
#     model.eval()
#     valid_loss = 0
#     correct = 0
#     for data, target in valid_loader:
#         data, target = Variable(data, volatile=True).cuda() , Variable(target).cuda()
#         output = model(data)
#         valid_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#
#     valid_loss /= len(valid_loader.dataset)
#     print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         valid_loss, correct, len(valid_loader.dataset),
#         100. * correct / len(valid_loader.dataset)))
#     losses.append(valid_loss)
#     return correct / len(valid_loader.dataset)

def to_img(x):
    x = 0.5 * (x+1)
    x = x.clamp(0,1)
    x = x.view(x.size(0),1,28,28)
    return x

def test(model, test_loader):
    model.eval()
    test_loss = 0
    #correct = 0
    for data, _ in test_loader:
        data = Variable(data, volatile=True).cuda()
        output = model.forward(data)
        test_loss += nn.MSELoss()(output, data).data[0] # sum up batch loss
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/test_imageOutput.png')
        pic = to_img(data.cpu().data)
        save_image(pic, './dc_img/test_imageInput_{}.png')
        #pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print('\n' + "test" + ' set: Average loss: {:.4f}\n'.format(test_loss))

def experiment(model, epochs=nombre_epochs, lr=le_ra):
    #best_precision = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model = train(model, train_loader, optimizer, epoch)
        print("epoch [{}/{}]".format(epoch, nombre_epochs))
        #precision = valid(model, valid_loader)
        #if precision > best_precision:
            #best_precision = precision
            #best_model = model
    # plt.plot(np.arange(0, epochs), losses)
    # plt.title("Log de vraisemblance pour 2 couches cachees avec dropout seulement a l'entree\nde 1000 noeuds chacune, avec 2 couches de convolution, \n{} epochs et LR = {}".format(epochs, lr))
    # plt.xlabel("Epochs")
    # plt.ylabel("Log de vraisemblance")
    # plt.show()
    #return best_model, best_precision

# Main
best_precision = 0
for model in [FcNetwork()] :
    model.cuda()
    experiment(model)
    # if precision > best_precision:
    #     best_precision = precision
    #     best_model = model

test(model, test_loader)
