import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import argparse

parser= argparse.ArgumentParser(description='SegNet segmentation')
parser.add_argument('--debug', type=bool, default=False,
                    choices=[True, False])
parser.add_argument('--network', type=bool, default=False,
                    choices=[True, False])

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        ## 13 vgg conv layer
        self.ConvEnc1= nn.Sequential(
                        nn.Conv2d(3,64,3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64,64,3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2, stride=2,return_indices=True),
        )
        self.ConvEnc2= nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,3,padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(2, stride=2,return_indices=True)
        )
        self.ConvEnc3= nn.Sequential(
                        nn.Conv2d(128,256,3,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,3,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,3,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.MaxPool2d(2, stride=2,return_indices=True)
        )
        self.ConvEnc4= nn.Sequential(
                        nn.Conv2d(256,256,3,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,512,3,padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,3,padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.MaxPool2d(2, stride=2,return_indices=True)
        )
        self.ConvEnc5= nn.Sequential(
                        nn.Conv2d(512,512,3,padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,3,padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,3,padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.MaxPool2d(2, stride=2,return_indices=True)
        )

    def forward(self,x, debug):
        print('>     Encoder')
        
        x1,indices1= self.ConvEnc1(x)
        if debug: print(x1.size())
        x2,indices2= self.ConvEnc2(x1)
        if debug: print(x2.size())
        x3,indices3= self.ConvEnc3(x2)
        if debug: print(x3.size())
        x4,indices4= self.ConvEnc4(x3)
        if debug: print(x4.size())
        x5, indices5= self.ConvEnc5(x4)
        if debug: print(x5.size())

        return [[x1,indices1],[x2,indices2],[x3, indices3],[x4,indices4],[x5, indices5]]

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        
        self.ConvDec1= nn.Sequential(
                        nn.ConvTranspose2d(512,512,3,padding=1),
                        nn.BatchNorm2d(512),
                        nn.ConvTranspose2d(512,512,3,padding=1),
                        nn.BatchNorm2d(512),
                        nn.ConvTranspose2d(512,512,3,padding=1),
                        nn.BatchNorm2d(512),
                        
        )
        self.ConvDec2= nn.Sequential(
                        nn.ConvTranspose2d(512,512,3,padding=1),
                        nn.BatchNorm2d(512),
                        nn.ConvTranspose2d(512,256,3,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ConvTranspose2d(256,256,3,padding=1),
                        nn.BatchNorm2d(256),
            
        )
        self.ConvDec3= nn.Sequential(
                        nn.ConvTranspose2d(256,256,3,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ConvTranspose2d(256,256,3,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ConvTranspose2d(256,128,3,padding=1),
                        nn.BatchNorm2d(128),
                        
        )
        self.ConvDec4= nn.Sequential(
                        nn.ConvTranspose2d(128,128,3,padding=1),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,64,3,padding=1),
                        nn.BatchNorm2d(64),            
                        
        )        
        self.ConvDec5= nn.Sequential( 
                        nn.ConvTranspose2d(64,64,3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ConvTranspose2d(64,10,3, padding=1),            
        )

    def forward(self, maxpool_list, debug):
        print('>     Decoder')
        x= self.ConvDec1(F.max_unpool2d(maxpool_list[4][0],indices=maxpool_list[4][1], kernel_size=2))
        if debug: print(x.size())
        #print((maxpool_list[3][0]+x).size())
        x= self.ConvDec2(F.max_unpool2d(x, indices=maxpool_list[3][1], kernel_size=2))
        if debug: print(x.size())
        x= self.ConvDec3(F.max_unpool2d( x, indices=maxpool_list[2][1], kernel_size=2))
        if debug: print(x.size())
        x= self.ConvDec4(F.max_unpool2d( x, indices=maxpool_list[1][1], kernel_size=2))
        if debug: print(x.size())
        x= self.ConvDec5(F.max_unpool2d( x, indices=maxpool_list[0][1], kernel_size=2))
        if debug: print(x.size())


class Segnet(nn.Module):
    def __init__(self, Encoder, Decoder, debug=True):
        super(Segnet, self).__init__()
        self.debug = debug
        # Encoder
        self.Encoder = Encoder()
        # decoder
        self.Decoder = Decoder()

    def forward(self, x):
        print('>> SegNet: ')
        maxpool_indices_list= self.Encoder(x, self.debug)
        
        segmented= self.Decoder(maxpool_indices_list,self.debug)


if __name__ == "__main__":
    args= parser.parse_args()
    net= Segnet(Encoder, Decoder, debug= args.debug)
    img= torch.randn(1,3,224,224)
    net(img)
    if args.network:
        print('>     Model')
        print(net)
