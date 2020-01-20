# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 19:12:44 2019

@author: shuan
"""

import sys
import os
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Definition of the module that is composed by a convolutional layer and a ReLU
def conv_relu(in_channel, out_channel, kernel, stride = 1, padding = 0):
    layer = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel, stride, padding),
            nn.ReLU(True)
            )
    return layer

# Mid-term network structure of branches
class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out1_2, out2_1, out2_2, out3_1, out3_2):
        super(inception, self).__init__()
        
        self.branch1 = nn.Sequential(
                conv_relu(in_channel, out1_1, 3, padding = 1),
                conv_relu(out1_1, out1_2, 3, padding = 1))
        
        self.branch2 = nn.Sequential(
                nn.AvgPool3d(2, stride = 2),
                conv_relu(in_channel, out2_1, 3, padding = 1),
                conv_relu(out2_1, out2_2, 3, padding = 1),
                nn.Upsample(scale_factor = 2, mode = 'trilinear'))
        
        self.branch3 = nn.Sequential(
                nn.AvgPool3d(2, stride = 2),
                nn.AvgPool3d(2, stride = 2),
                conv_relu(in_channel, out3_1, 3, padding = 1),
                conv_relu(out3_1, out3_2, 3, padding = 1),
                nn.Upsample(scale_factor = 4 , mode = 'trilinear'))
        
    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        output = f1 + f2 + f3
        return output

# Overall Network Sturcture
class FluidNet(nn.Module):
    def __init__(self, in_channel, mid_channel, verbose = False):
        super(FluidNet, self).__init__()
        self.verbose = verbose
        
        self.block1 = nn.Sequential(
                conv_relu(in_channel, out_channel = mid_channel, kernel = 3, padding = 1))
        
        self.block2 = nn.Sequential(
                inception(mid_channel, mid_channel, mid_channel, mid_channel, mid_channel, mid_channel, mid_channel),
                conv_relu(mid_channel, mid_channel, kernel = 1),
                conv_relu(mid_channel, 1, kernel = 1),
                )
        
    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block output: {}'.format(x.shape))
            
        x = self.block2(x)
        if self.verbose:
            print('block output: {}'.format(x.shape))
            
        return x

# Conversion of int to string
def int_to_str_fillzero(val):
    if val < 0 or val >= 1000000:
        print('Out of range!')
        return False
    elif val < 10:
        return '00000' + str(val)
    elif val < 100:
        return '0000' + str(val)
    elif val < 1000:
        return '000' + str(val)
    elif val < 10000:
        return '00' + str(val)
    elif val < 100000:
        return '0' + str(val)
    else:
        return str(val)


# Dataloader of the overall network
class FluidDataSet2(Dataset):
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False):
        super(FluidDataSet2, self).__init__()
        self.root = root
    
    def __getitem__(self, index):
        index1 = int(index / 64)
        index2 = (index - index1 * 64) * 4
        file_root = self.root + int_to_str_fillzero(index1) + '/' + int_to_str_fillzero(index2)
        
        geom = open(file_root + '_flags.txt')
        rdline_geom = geom.readlines()
        channel_geom = np.array([float(rdline_geom[i][:-1]) for i in range(len(rdline_geom))])
        channel_geom.resize(1, 64, 64, 64)
        channel1 = torch.from_numpy(channel_geom)
        geom.close()
        
        UDiv = open(file_root + '_UDiv.txt')
        rdline_UDiv = UDiv.readlines()
        channel_UDiv = np.array([float(rdline_UDiv[i][:-1]) for i in range(len(rdline_UDiv))])
        channel_UDiv.resize(1, 64, 64, 64)
        channel2 = torch.from_numpy(channel_UDiv)
        UDiv.close()
        
        input_channel = torch.cat((channel1, channel2), 0)
        
        pDiv = open(file_root + '_p.txt')
        rdline_pDiv = pDiv.readlines()
        channel_pDiv = np.array([float(rdline_pDiv[i][:-1]) for i in range(len(rdline_pDiv))])
        channel_pDiv.resize(1, 64, 64, 64)
        label = torch.from_numpy(channel_pDiv)
        pDiv.close()
        
        return input_channel, label
        
    def __len__(self):
        return 64 * len([lists for lists in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, lists))])

def get_variable(x, y = False):
    x = Variable(x, requires_grad = y)
    return x.cuda() if torch.cuda.is_available() else x

# Codes for training and testing the overall CNN with given parameters
def model_train_test(model, train_dataloader, test_dataloader, epoch, optimizer, criterion):
   
    for epoch_idx in range(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.cuda(), target.cuda()
            model.cuda()
            data, target = get_variable(data.float()), get_variable(target.float())
            
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.data
           
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 4 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_idx, (batch_idx+1) * len(data), len(train_dataloader.dataset),
                100. * (batch_idx+1) / len(train_dataloader), loss.data))
                
                '''
                for name, param in model.named_parameters():
                    if (name == 'block2.2.0.bias') or (name == 'block2.1.0.bias'):
                        print(name, param.grad)
                    if param.requires_grad:
                        print(name)
                '''
            torch.cuda.empty_cache()
                
        train_loss /= len(train_dataloader)
        print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
        
        model.eval()
        test_loss = 0
        for batch_idx, (data, target) in enumerate(test_dataloader):
            model.cuda()
            data, target = get_variable(data.float()), get_variable(target.float())
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.data
            
            if (batch_idx + 1) % 4 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_idx, (batch_idx+1) * len(data), len(test_dataloader.dataset),
                100. * (batch_idx+1) / len(test_dataloader), loss.data))   
            
            torch.cuda.empty_cache()
            
        test_loss /= len(test_dataloader)
        print('\nTest set: Average loss: {:.4f}'.format(test_loss))

# Training codes
if __name__ == '__main__':
    train_dataset = FluidDataSet2(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/data/tr_Test_data/')
    trainloader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
    test_dataset = FluidDataSet2(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/data/te_Test_data/')
    testloader = DataLoader(test_dataset, batch_size = 4, shuffle = True)
    train_network = FluidNet(2, 8)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(train_network.parameters(), lr = 0.001)
    criterion.cuda()

    model_train_test(train_network, trainloader, testloader, 3, optimizer, criterion)



    
    