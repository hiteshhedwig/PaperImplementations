import torch 
import torch.nn as nn
import numpy as np

class Separable(nn.Module):
    def __init__(self, channels, middle_flow= False):
        super(Separable, self).__init__()
        
        if middle_flow== False: 
        
            self.block= nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(channels[0], channels[0],3, groups=channels[0], padding=1),
                nn.Conv2d(channels[0], channels[1],1),
                nn.BatchNorm2d( channels[1]),
                nn.ReLU(),
                nn.Conv2d( channels[1], channels[1],3, groups=channels[1], padding=1),
                nn.Conv2d( channels[1], channels[1],1),
                nn.BatchNorm2d(channels[1]),
                nn.MaxPool2d(3,stride=2, padding=1),
            )
        
        if middle_flow==True:
            self.block= nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(channels[0], channels[0],3, groups=channels[0], padding=1),
                nn.Conv2d(channels[0], channels[1],1),
                nn.BatchNorm2d( channels[1]),
                nn.ReLU(),
                nn.Conv2d( channels[1], channels[1],3, groups=channels[1], padding=1),
                nn.Conv2d( channels[1], channels[1],1),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(),
                nn.Conv2d( channels[1], channels[1],3, groups=channels[1], padding=1),
                nn.Conv2d( channels[1], channels[1],1),                
            )

        if middle_flow=='Exit':
            self.block= nn.Sequential(
                nn.Conv2d(channels[0], channels[0],3, groups=channels[0], padding=1),
                nn.Conv2d(channels[0], channels[1],1),
                nn.BatchNorm2d( channels[1]),
                nn.ReLU(),
                nn.Conv2d( channels[1], channels[1],3, groups=channels[1], padding=1),
                nn.Conv2d( channels[1], channels[2],1),
                nn.BatchNorm2d(channels[2]),
                nn.ReLU(),
            )


    
    def forward(self,x):
        x= self.block(x)
        return x


class Xception(nn.Module):
    def __init__(self, Separable):
        super(Xception, self).__init__()

        self.conv1_3x3 = nn.Conv2d(3,32,3,stride=2,padding=1)
        self.bn1 =nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.conv2_3x3= nn.Conv2d(32,64,3,stride=1,padding=1)
        self.bn2 =nn.BatchNorm2d(64)
        self.relu= nn.ReLU()

        self.block_3x3= Separable(channels=[64,128])
        self.downsample_1= nn.Conv2d(64,128,1,stride=2)

        self.block_3x3_256= Separable(channels=[128,256])
        self.downsample_2= nn.Conv2d(128,256,1,stride=2)

        self.block_3x3_728= Separable(channels=[256,728])
        self.downsample_3= nn.Conv2d(256,728,1,stride=2)

        self.block_3x3_middle= Separable(channels=[728,728], middle_flow=True)

        self.block_3x3_exit = Separable(channels=[728,1024])
        self.downsample_4= nn.Conv2d(728,1024,1,stride=2)

        self.block_3x3_exit_2 = Separable(channels=[1024,1536,2048], middle_flow='Exit')
        
        self.avgpool= nn.AdaptiveAvgPool2d((1,1))

        self.fc= nn.Linear(2048,1000)

    def forward(self,x):
        x= self.relu(self.bn1(self.conv1_3x3(x)))
        print(x.size())
        x_identity = self.relu(self.bn2(self.conv2_3x3(x)))
        print(x_identity.size())
        x_res_1= self.block_3x3(x_identity)
        print(x_res_1.size())
        x_identity= self.downsample_1(x_identity)
        print(x_identity.size())
        
        x_identity= x_identity+x_res_1

        x_res_2= self.block_3x3_256(x_identity)
        print(x_res_2.size())
        x_identity= self.downsample_2(x_identity)
        print(x_identity.size())
        x_identity= x_identity+x_res_2
    
        x_res_3= self.block_3x3_728(x_identity)
        print(x_res_3.size())
        x_identity= self.downsample_3(x_identity)
        print(x_identity.size())
        x_identity= x_identity+x_res_3

        for i in range(9):
            x_res_4= self.block_3x3_middle(x_identity)
            x_identity= x_identity+x_res_4

        print(x_identity.size())
        
        x_res_5= self.block_3x3_exit(x_identity)
        x_identity= self.downsample_4(x_identity)

        x_identity= x_identity+x_res_5
        print(x_identity.size())

        x_res_6= self.block_3x3_exit_2(x_identity)
        print(x_res_6.size())

        x= self.avgpool(x_res_6)
        print(x.size())

        x= x.view(x.shape[0],-1)
        print(x.size())

        x= self.fc(x)
        print(x.size())       
        

if __name__ == "__main__":
    img= torch.randn(1,3,299,299)

    net= Xception(Separable)
    net(img)
    #print(net)