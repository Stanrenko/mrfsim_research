
#import matplotlib
#matplotlib.use("TkAgg")
from mrfsim import T1MRF
from image_series import *
from dictoptimizers import SimpleDictSearch
from utils_mrf import *
import json
import readTwix as rT
import time
import os
from numpy.lib.format import open_memmap
from numpy import memmap
import pickle
from scipy.io import loadmat,savemat

import torch
from torch import nn
from torch.utils.data import DataLoader
from skimage import io

base_folder = "./data/InVivo/3D/"

X=np.load("X_movement_from_nav.npy")
Y=np.load("Y_movement_from_nav.npy")

X=X.reshape(10000,50,-1,50)
X=np.moveaxis(X,1,2)
X=np.moveaxis(X,-1,-2)

num=0
ch=6
plt.figure()
plt.imshow(X[num,ch,:,:])
plt.figure()
plt.plot(Y[num])

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''

    def __init__(self, input_channels):
        super(ContractingBlock, self).__init__()
        # You want to double the number of channels in the first convolution
        # and keep the same number of channels in the second.
        #### START CODE HERE ####
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3)

        #self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(input_channels * 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #### END CODE HERE ####

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock:
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        x = self.activation(x)
        x=self.bn(x)
        #x = self.conv2(x)
        #x = self.activation(x)
        #x = self.maxpool(x)
        return x

    # Required for grading
    def get_self(self):
        return self


class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a UNet -
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        # "Every step in the expanding path consists of an upsampling of the feature map"
        #### START CODE HERE ####
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        #### END CODE HERE ####

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock:
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Encoder, self).__init__()
        # "Every step in the expanding path consists of an upsampling of the feature map"
        self.hidden_channels=hidden_channels
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.final_layer_activation=nn.ReLU(inplace=True)
        #self.contract4 = ContractingBlock()
        self.fc1 = nn.Linear(hidden_channels * 2*48*48, 4*output_channels)
        self.fc_bn=nn.BatchNorm1d(4*output_channels)
        self.fc2 = nn.Linear(4*output_channels, output_channels)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        '''
        Function for completing a forward pass of UNet:
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        # Keep in mind that the expand function takes two inputs,
        # both with the same number of channels.
        #### START CODE HERE ####
        #print(x.shape)
        x = self.upfeature(x)
        #print(x.shape)
        x = self.contract1(x)
        #print(x.shape)
        #x2 = self.contract2(x1)
        #print(x2.shape)
        #x3 = self.contract3(x2)
        #print(x3.shape)
        #x4 = self.contract4(x3)
        x = x.view(-1,self.hidden_channels*2*48*48)
        x=self.fc1(x)
        x=self.dropout(x)
        x = self.final_layer_activation(x)
        x=self.fc_bn(x)
        x=self.fc2(x)
        x = self.dropout(x)
        x=self.final_layer_activation(x)
        #print(xn.shape)
        #### END CODE HERE ####
        return x


volumes = torch.Tensor(X)
labels = torch.Tensor(Y)
dataset = torch.utils.data.TensorDataset(volumes, labels)

criterion = nn.MSELoss()
n_epochs = 1500
input_dim =volumes.shape[1]
label_dim = labels.shape[1]
batch_size = 64
lr = 0.001
device = 'cuda'
display_step=19

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size


train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


def train():
    losses=[]
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)
    unet = Encoder(input_dim, label_dim,64).to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(unet_opt, 'min',patience=50)
    cur_step = 0
    current_loss=np.inf
    for epoch in range(n_epochs):
        batch_loss=[]
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            #print(real.shape)
            # Flatten the image
            real = real.to(device)
            labels = labels.to(device)

            ### Update U-Net ###
            unet_opt.zero_grad()
            pred = unet(real)
            #print(pred.shape)
            unet_loss = criterion(pred, labels)
            unet_loss.backward()
            unet_opt.step()

            #if cur_step % display_step == 0:
            #    print(f"Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")

            batch_loss.append(unet_loss.item())
            cur_step += 1
        mean_loss=np.mean(batch_loss)
        print(f"Epoch {epoch}: U-Net loss: {mean_loss}")
        losses.append(mean_loss)
        if mean_loss<current_loss:
            current_loss=mean_loss
            torch.save(unet,"model_3.pt")

        scheduler.step(current_loss)
        print(unet_opt.param_groups[0]['lr'])

    return unet,np.array(losses)

unet,losses=train()
np.save("model_2_losses.npy",losses)
plt.figure()
plt.plot(losses)


unet=torch.load("model_2.pt")

num=np.random.choice(range(len(test_dataset)))




with torch.no_grad():
    pred = unet(test_dataset[num][0][None,:,:,:].to(device))


len(test_dataset)

plt.figure()
plt.plot(test_dataset[num][1])
plt.plot(pred.cpu().numpy().squeeze())