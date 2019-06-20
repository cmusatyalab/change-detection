import torch.nn as nn
import torch
import pdb
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from scipy import ndimage
import torchvision.models as M
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models 


from pdb import set_trace as st

from collections import OrderedDict

#import utils

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
''' #################

BEGIN UNET

''' #################
vgg16_layer_list = ["features.0.weight", "features.0.bias", "features.2.weight", 
"features.2.bias", "features.5.weight", "features.5.bias", "features.7.weight", 
"features.7.bias", "features.10.weight", "features.10.bias", "features.12.weight", 
"features.12.bias", "features.14.weight", "features.14.bias", "features.17.weight", 
"features.17.bias", "features.19.weight", "features.19.bias", "features.21.weight", 
"features.21.bias", "features.24.weight", "features.24.bias", "features.26.weight", 
"features.26.bias", "features.28.weight", "features.28.bias"]
class ResidualBlock2(nn.Module):
    def __init__(self, channels_in, channels_out=64):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
    def forward(self, input_x):
        x = self.conv1(input_x)
        x = self.conv2(x)
        return x

class ResidualBlock3(nn.Module):
    def __init__(self, channels_in, channels_out=64):
        super(ResidualBlock3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
    def forward(self, input_x):
        x = self.conv1(input_x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class DecodeBlock4(nn.Module):
    def __init__(self, channels_in, channels_out=64):
        super(DecodeBlock4, self).__init__()
        self.conv1 =nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        self.conv2 =nn.Sequential(
            nn.Conv2d(2*channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        self.conv3 =nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        self.conv4 =nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        
    def forward(self, input_x, concat_x):
        x = self.conv1(input_x)
        x = torch.cat((x, concat_x),dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class DecodeBlock3(nn.Module):
    def __init__(self, channels_in, channels_out=64):
        super(DecodeBlock3, self).__init__()
        self.conv1 =nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        self.conv2 =nn.Sequential(
            nn.Conv2d(2*channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        self.conv3 =nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        self.conv4 =nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        
    def forward(self, input_x, concat_x):
        x = self.conv1(input_x)
        x = torch.cat((x, concat_x),dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Encoder(nn.Module):
    def __init__(self, channels_in=3, channels_out=1024):
        super(Encoder, self).__init__()
        self.convblock1 = ResidualBlock2(3, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.convblock2 = ResidualBlock2(64, 128)
        self.convblock3 = ResidualBlock3(128, 256)
        self.convblock4 = ResidualBlock3(256, 512)
        self.convblock5 = ResidualBlock3(512, 512)
        self.dropout = nn.Dropout()
        self.convblock6 = ResidualBlock2(512, channels_out)
        

    def forward(self, input_x):
        x0 = self.convblock1(input_x)
        x1 = self.maxpool(x0)
        x2 = self.convblock2(x1)
        x3 = self.maxpool(x2)
        x4 = self.convblock3(x3)
        x5 = self.maxpool(x4)
        x6 = self.convblock4(x5)
        x7 = self.maxpool(x6)
        x8 = self.convblock5(x7)
        x8 = self.dropout(x8)
        x9 = self.convblock6(x8)
        x9 = self.dropout(x9)
        return x9, x8, x6, x4, x2, x0
        '''
        x9 = torch.Size([1, 1024, 14, 14])
        x8 = torch.Size([1, 512, 14, 14])
        x6 = torch.Size([1, 512, 28, 28])
        x4 = torch.Size([1, 256, 56, 56])
        x2 = torch.Size([1, 128, 112, 112])
        x0 = torch.Size([1, 64, 224, 224])
        '''

class Decoder(nn.Module):
    def __init__(self, channels_in=1024, channels_out=10):
        super(Decoder, self).__init__()
        self.convblock1 = DecodeBlock3(channels_in, 512)
        self.convblock2 = DecodeBlock4(512, 512)
        self.convblock3 = DecodeBlock4(512, 256)
        self.convblock4 = DecodeBlock4(256, 128)
        self.convblock5 = DecodeBlock4(128, 64)
        self.convblock6 = nn.Sequential(
            nn.Conv2d(64, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
            )
    def forward(self, x9, x8, x6, x4, x2, x0):
        x = self.convblock1(x9,x8)
        x = self.convblock2(x,x6)
        x = self.convblock3(x,x4)
        x = self.convblock4(x,x2)
        x = self.convblock5(x,x0)
        x = self.convblock6(x)
        return x

class UNet(nn.Module):
    """
    UNet for change detection
    """
    def __init__(self, batch_norm=False, channels_in=3, channels_out=10):
        super().__init__()
        self.encoder = Encoder(channels_in, 1024)
        self.decoder1 = Decoder(1024,channels_out)
        self.decoder2 = Decoder(1024,channels_out)
        self.finalconv = nn.Sequential(
            nn.Conv2d(2*channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
            )
        self._initialize_weights('Xavier')

        if batch_norm:
            pretrained_state = model_zoo.load_url(model_urls['vgg16_bn'])
        else:
            pretrained_state = model_zoo.load_url(model_urls['vgg16'])
        #self.encoder.load_state_dict()

        own_state = self.encoder.state_dict()
        #pretrained_state = {key:value for key,value in pretrained_state.items() if key.split('.')[0] == "features"}
        # own_state.update(pretrained_state)
        # self.encoder.load_state_dict(own_state)
        own_name_list = list(own_state.keys())
        # print(own_name_list)
        # print(list(pretrained_state.keys()))
        k=0
        for i, (name, param) in enumerate(pretrained_state.items()):
            #print(name)
            if isinstance(param, nn.Parameter):
                param = param.data
            if 'features' in name:
                try:
                    if 'num_batches_tracked' in own_name_list[k]:
                        k+=1
                    own_state[own_name_list[k]].copy_(param)
                    print('Copied data from {} to {}'.format(name, own_name_list[k]))
                    # print(param.shape)
                    # print(own_state[own_name_list[k]].shape)
                except:
                    print('Failed: Copy {} to {}'.format(name, own_name_list[k]))
                    # print(param.shape)
                    # print(own_state[own_name_list[k]].shape)
                    return
            k+=1
        
        # for i, (name, param) in enumerate(own_state.items()):
        #     print(name)
            # # it is not loading the classifier portion due to number mismatch, so change the name so it loads
            # #if 'features' in name:
            # name = unet_layer_list[i]
            # if name not in own_state:
            #     continue
            # if isinstance(param, nn.Parameter):
            #     param = param.data
            # try:
            #     own_state[name].copy_(param)
            #     print('Copied {}'.format(name))
            # except:
            #     print('Did not find {}'.format(name))
            #     continue

    def _initialize_weights(self, method):
        if method == 'Xavier':
            print('Initializing with Xavier\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        elif method == 'He':
            print('Initializing with He\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        return

    def forward(self, img1, img2):
        """ Input:
            img1 (batch_sizex3x224x224)
            img2 (batch_sizex3x224x224)
            out 
        """
        x9, x8, x6, x4, x2, x0 = self.encoder(img1)
        out1 = self.decoder1(x9, x8, x6, x4, x2, x0)
        x9, x8, x6, x4, x2, x0 = self.encoder(img2)
        out2 = self.decoder2(x9, x8, x6, x4, x2, x0)
        combined = torch.cat((out1, out2),dim=1)
        #combined = out2-out1
        final = self.finalconv(combined)
        return final

class UNetSeg(nn.Module):
    """
    UNet for segmentation
    """
    def __init__(self, channels_in=3, channels_out=10):
        super().__init__()
        self.encoder = Encoder(channels_in, 1024)
        self.decoder = Decoder(1024,channels_out)
        self.finalconv = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
            )
        self._initialize_weights('He')
        pretrained_state = model_zoo.load_url(model_urls['vgg16'])
        #self.encoder.load_state_dict()

        own_state = self.encoder.state_dict()
        
        own_name_list = list(own_state.keys())
        for i, (name, param) in enumerate(pretrained_state.items()):
            if isinstance(param, nn.Parameter):
                param = param.data
            if 'features' in name:
                try:
                    own_state[own_name_list[i]].copy_(param)
                    print('Copied data from {} to {}'.format(name, own_name_list[i]))
                except:
                    print('Did not find {}'.format(name))
                    continue
        

    def _initialize_weights(self, method):
        if method == 'Xavier':
            print('Initializing with Xavier\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        elif method == 'He':
            print('Initializing with He\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        return

    def forward(self, img):
        """ Input:
            img (batch_sizex3x224x224)
            out 
        """
        x9, x8, x6, x4, x2, x0 = self.encoder(img)
        out = self.decoder(x9, x8, x6, x4, x2, x0)
        
        final = self.finalconv(out)
        return final


class ChangeNetEncoder(nn.Module):
    """
    ChangeNet Encoder- ResNet-50
    """
    def __init__(self):
        super().__init__()
        self.res50_model = models.resnet50(pretrained=True)
        ### ResNet-50 Encoder ###

        # Extracts 512 x 28 x 28
        self.res50_3 = nn.Sequential(*list(self.res50_model.children())[:-4])
        # Extracts 1024 x 14 x 14
        self.res50_4 = nn.Sequential(*list(self.res50_model.children())[-4:-3])
        # Extracts 2048 x 7 x 7
        self.res50_5 = nn.Sequential(*list(self.res50_model.children())[-3:-2])
        for child in self.res50_model.children():
            print(child)
        print(len(list(self.res50_model.children())))

    def forward(self, img):
        """ Input:
            img (batch_size x 3 x 224 x 224)
        """
        x3 = self.res50_3(img)     
        x4 = self.res50_4(x3)
        x5 = self.res50_5(x4)
        return x5, x4, x3

class ChangeNetDecoder(nn.Module):
    """
    ChangeNet Encoder- ResNet-50
    """
    def __init__(self, channels_out=10):
        super().__init__()
        ### Decoder ###
        self.fc3 = nn.Sequential(
            nn.Conv2d(512, channels_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
            )
        self.fc4 = nn.Sequential(
            nn.Conv2d(1024, channels_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
            )
        self.fc5 = nn.Sequential(
            nn.Conv2d(2048, channels_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
            )
        self.deconv3 =nn.Sequential(
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU()
            )
        self.deconv4 =nn.Sequential(
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU()
            )
        self.deconv5 =nn.Sequential(
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU()
            )

        self._initialize_weights('He')

    def _initialize_weights(self, method):
        if method == 'Xavier':
            print('Initializing with Xavier\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        elif method == 'He':
            print('Initializing with He\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        return

    def forward(self, x5, x4, x3):
        """ Input:
        """
        x3 = self.fc3(x3)
        x3 = self.deconv3(x3)

        x4 = self.fc4(x4)
        x4 = self.deconv4(x4)

        x5 = self.fc5(x5) 
        x5 = self.deconv5(x5)   
        
        return x5, x4, x3

class ConcatAndFC(nn.Module):
    def __init__(self, channels_out=10):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*channels_out, channels_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2*channels_out, channels_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(2*channels_out, channels_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
            )
        self._initialize_weights('He')

    def _initialize_weights(self, method):
        if method == 'Xavier':
            print('Initializing with Xavier\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        elif method == 'He':
            print('Initializing with He\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        return
    
    def forward(self, x5,y5,x4,y4,x3,y3):
        x5 = torch.cat((x5, y5),dim=1)
        x4 = torch.cat((x4, y4),dim=1)
        x3 = torch.cat((x3, y3),dim=1)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x5)
        x = x3+x4+x5
        return x

class ChangeNet(nn.Module):
    """
    ChangeNet Third-party implmenetation based on paper: 
    ChangeNet: A Deep Learning Architecture for Visual Change Detection
    """
    def __init__(self, channels_in=3, channels_out=10):
        super().__init__()
        
        self.encoder = ChangeNetEncoder()
        self.decoder1 = ChangeNetDecoder()
        self.decoder2 = ChangeNetDecoder()
        self.fc = ConcatAndFC()
        # for param in self.encoder.parameters():
        #     param.requires_grad = False


    def forward(self, img1, img2):
        """ Input:
            img (batch_sizex3x224x224)
            out 
        """
        x5, x4, x3 = self.encoder(img1)
        y5, y4, y3 = self.encoder(img2)
        x5, x4, x3 = self.decoder1(x5,x4,x3)
        y5, y4, y3 = self.decoder2(y5,y4,y3)
        x = self.fc(x5,y5,x4,y4,x3,y3)
        return x

class ResNetEncoder(nn.Module):
    """
    ChangeNet Encoder- ResNet-50
    """
    def __init__(self):
        super().__init__()
        #self.res50_model = models.resnet50(pretrained=True)
        self.res50_model = models.resnet50(pretrained=True)
        ### ResNet-50 Encoder ###
        # Extracts 64 x 112 x 112
        self.res50_0 = nn.Sequential(*list(self.res50_model.children())[:-9])
        # Extracts 64 x 112 x 112
        # self.res50_1 = nn.Sequential(*list(self.res50_model.children())[-9:-8])
        # # Extracts 64 x 112 x 112
        # self.res50_2 = nn.Sequential(*list(self.res50_model.children())[-8:-7])
        # Extracts 64 x 56 x 56
        self.res50_3 = nn.Sequential(*list(self.res50_model.children())[-9:-6])
        # # Extracts 512 x 56 x 56
        # self.res50_4 = nn.Sequential(*list(self.res50_model.children())[-6:-5])
        # Extracts 512 x 28 x 28
        self.res50_5 = nn.Sequential(*list(self.res50_model.children())[-6:-4])
        # Extracts 1024 x 14 x 14
        self.res50_6 = nn.Sequential(*list(self.res50_model.children())[-4:-3])
        # Extracts 2048 x 7 x 7
        self.res50_7 = nn.Sequential(*list(self.res50_model.children())[-3:-2])
        self.dropout = nn.Dropout()
        for child in self.res50_model.children():
            print(child)

    def forward(self, img):
        """ Input:
            img (batch_size x 3 x 224 x 224)
        """
        x0 = self.res50_0(img)
        #x1 = self.res50_1(x0)
        #x2 = self.res50_2(x1)
        x3 = self.res50_3(x0)     
        #x4 = self.res50_4(x3)
        x5 = self.res50_5(x3)
        x6 = self.res50_6(x5)
        x6 = self.dropout(x6)
        x7 = self.res50_7(x6)
        x7 = self.dropout(x7)
        return x7, x6, x5, x3, x0

class ResNetDecoder(nn.Module):
    def __init__(self, channels_in=2048, channels_out=10):
        super(ResNetDecoder, self).__init__()
        self.convblock1 = DecodeBlock4(channels_in, 1024) # 14 x 14
        self.convblock2 = DecodeBlock4(1024, 512) # 28 x 28
        self.convblock3 = DecodeBlock4(512, 64) # 56 x 56
        self.convblock4 = DecodeBlock4(64, 64) # 112 x 112
        # self.convblock2 = DecodeBlock4(1024, 512) # 28 x 28
        # self.convblock3 = DecodeBlock4(512, 256) # 56 x 56
        # self.convblock4 = DecodeBlock4(256, 64) # 112 x 112
        
        self.convblock5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
            )
        self._initialize_weights('He')
    
    def _initialize_weights(self, method):
        if method == 'Xavier':
            print('Initializing with Xavier\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        elif method == 'He':
            print('Initializing with He\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        return

    def forward(self, x7, x6, x5, x3, x0):
        #pdb.set_trace()
        x = self.convblock1(x7,x6)
        x = self.convblock2(x,x5)
        x = self.convblock3(x,x3)
        x = self.convblock4(x,x0)
        x = self.convblock5(x)
        #x = self.convblock6(x)
        return x

class ResNetUNet(nn.Module):
    """
    UNet for change detection
    """
    def __init__(self, batch_norm=False, channels_in=3, channels_out=10):
        super().__init__()
        #self.encoder = resnet50mod(pretrained=True)
        self.encoder = ResNetEncoder()
        self.decoder1 = ResNetDecoder()
        self.decoder2 = ResNetDecoder()
        self.finalconv = nn.Sequential(
            nn.Conv2d(2*channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        self._initialize_weights('He')
        
    def _initialize_weights(self, method):
        if method == 'Xavier':
            print('Initializing with Xavier\'s Method')
            for m in self.finalconv.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        elif method == 'He':
            print('Initializing with He\'s Method')
            for m in self.finalconv.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        return

    def forward(self, img1, img2):
        """ Input:
            img1 (batch_sizex3x224x224)
            img2 (batch_sizex3x224x224)
            out 
        """
        x7, x6, x5, x3, x0 = self.encoder(img1)
        out1 = self.decoder1(x7, x6, x5, x3, x0)
        x7, x6, x5, x3, x0 = self.encoder(img2)
        out2 = self.decoder2(x7, x6, x5, x3, x0)
        combined = torch.cat((out1, out2),dim=1)
        final = self.finalconv(combined)
        return final

def vgg_preprocess(x_in):
    #z = 255.0 * (x_in + 1.0) / 2.0
    z = 255.0 * x_in
    z[:, 0, :, :] -= 103.939
    z[:, 1, :, :] -= 116.779
    z[:, 2, :, :] -= 123.68
    return z

class VGG19FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = M.vgg19(pretrained=True)
        self.features = nn.Sequential(
            *list(self.model.features.children())#[:9]
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, img):
        # If image is 0 to 1, we are fine
        return self.features(vgg_preprocess(img))

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss,self).__init__()
        
    def forward(self,y_pred, y_true, lmbda=1.0):
        return vgg_loss(y_pred, y_true, lmbda=lmbda)


def vgg_loss(y_pred, y_true, lmbda=1.0):
    return torch.mean(torch.sum(torch.abs(y_pred - y_true),dim=1))

class Discriminator(nn.Module):
    """
    Discriminator
    """
    def __init__(self, in_c=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            # Use BCEWithLogitsLoss so we don't need sigmoid layer
            #nn.Sigmoid()
        )

        self._initialize_weights('He')

    def _initialize_weights(self, method):
        if method == 'Xavier':
            print('Initializing with Xavier\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        elif method == 'He':
            print('Initializing with He\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        return

    def forward(self, img, mask):
        """ Input:
            img (batch_sizex3x256x256)
            mask (batch_sizex1x256x256)
        """
        # Concatenate along channels
        #x = torch.cat((img, pose_src, pose_tgt), dim=1) #channels = 51
        x = torch.cat((img, mask), dim=1) #channels = 4
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
                
        return x

class UNetDown(nn.Module):
    def __init__(self, channels_in, channels_out, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(channels_in, channels_out, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(channels_out))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, channels_in, channels_out, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(channels_out),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, channels_in=3, channels_out=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(channels_in, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, channels_out, 4, padding=1),
            nn.Tanh(),
        )
        self._initialize_weights('He')

    def _initialize_weights(self, method):
        if method == 'Xavier':
            print('Initializing with Xavier\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        elif method == 'He':
            print('Initializing with He\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        return

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################


class Pix2PixDiscriminator(nn.Module):
    def __init__(self, channels_in=3):
        super(Pix2PixDiscriminator, self).__init__()
        
        def discriminator_block(channels_in, channels_out, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(channels_in, channels_out, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(channels_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels_in * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )
        self._initialize_weights('He')

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

    def _initialize_weights(self, method):
        if method == 'Xavier':
            print('Initializing with Xavier\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        elif method == 'He':
            print('Initializing with He\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        return

##############################
#        Modified ResNet
##############################

class ResNetNoBatchNorm(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNetNoBatchNorm, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.dropout = nn.Dropout()
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckNoBatch):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                #nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3 = self.dropout(x3)
        x4 = self.layer4(x3)
        x4 = self.dropout(x4)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x4, x3, x2, x1, x0

class BottleneckNoBatch(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckNoBatch, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        #self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        #out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def resnet50mod(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model, with no batch normalization.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    pretrained = models.resnet50(pretrained=True)
    model = ResNetNoBatchNorm(BottleneckNoBatch, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        own_state = model.state_dict()
        pretrained_state = pretrained.state_dict()
        pretrained_state = { key:value for key,value in pretrained_state.items() if key in own_state.keys() }
        #if "bn" not in key and "fc" not in key and "running_var" not in key and "batches" not in key}
        #print(own_state)
        own_state.update(pretrained_state)
        model.load_state_dict(own_state)
        
    return model

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class UNetInpainting(nn.Module):
    """
    UNet for segmentation
    """
    def __init__(self, channels_in=3, channels_out=3):
        super().__init__()
        self.encoder = Encoder(channels_in, 1024)
        self.decoder = Decoder(1024,channels_out)
        self.finalconv = nn.Sequential(
            nn.Conv2d(channels_in+channels_out, channels_out, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.ReLU()
            nn.Sigmoid()
            )
        self._initialize_weights('He')
        # pretrained_state = model_zoo.load_url(model_urls['vgg16'])
        # #self.encoder.load_state_dict()

        # own_state = self.encoder.state_dict()
        
        # own_name_list = list(own_state.keys())
        # for i, (name, param) in enumerate(pretrained_state.items()):
        #     if isinstance(param, nn.Parameter):
        #         param = param.data
        #     if 'features' in name:
        #         try:
        #             own_state[own_name_list[i]].copy_(param)
        #             print('Copied data from {} to {}'.format(name, own_name_list[i]))
        #         except:
        #             print('Did not find {}'.format(name))
        #             continue
        

    def _initialize_weights(self, method):
        if method == 'Xavier':
            print('Initializing with Xavier\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        elif method == 'He':
            print('Initializing with He\'s Method')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
        return

    def forward(self, img):
        """ Input:
            img (batch_sizex3x224x224)
            mask (batch_sizex3x224x224) 
        """
        #img = torch.cat((img, mask), dim=1) #channels = 4
        x9, x8, x6, x4, x2, x0 = self.encoder(img)
        out = self.decoder(x9, x8, x6, x4, x2, x0)
        out = torch.cat((img,out),dim=1)
        final = self.finalconv(out)
        return final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Networks')
    parser.add_argument('--im', type=str, default='./')
    args = parser.parse_args()
    #model = ResNetEncoder().cuda()
    model = resnet50mod(pretrained=True).cuda()
    img = np.ones((1,3,224,224), dtype=np.float32)
    img = torch.from_numpy(img).cuda(async=True)
    x4, x3, x2, x1, x0 = model(img)
    print(x4.shape)
    print(x3.shape)
    print(x2.shape)
    print(x1.shape)
    print(x0.shape)

    
    # img = Image.open(args.imPath)
    # img = transforms.ToTensor()(img.convert('RGB'))
    # img = img.cuda(async=True)
    
    '''
    img = np.ones((1,3,224,224), dtype=np.float32)
    img = torch.from_numpy(img).cuda(async=True)

    mask = np.ones((1,1,224,224), dtype=np.float32)
    mask = torch.from_numpy(mask).cuda(async=True)

    pose = np.ones((1,24,224,224), dtype=np.float32)
    pose = torch.from_numpy(pose).cuda(async=True)

    model = UNet().cuda()
    print(model)
    model.eval()
    x = model(img,img)
    print(x.shape)
    '''
