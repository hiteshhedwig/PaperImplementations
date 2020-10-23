import torch
import torch.nn as nn
import numpy as np

class Incep(nn.Module):
    def __init__(self, channels_list):
        super(Incep,self).__init__()
        self.conv_incep1= nn.Conv2d(channels_list[0],channels_list[1],kernel_size=1)
        self.relu= nn.ReLU()
        self.conv_incep2= nn.Conv2d(channels_list[0],channels_list[2], kernel_size=1)
        self.relu1= nn.ReLU()
        self.conv_incep2_1= nn.Conv2d(channels_list[2],channels_list[3],kernel_size=3, padding=1)
        self.relu2= nn.ReLU()
        self.conv_incep3= nn.Conv2d(channels_list[0],channels_list[4],kernel_size=1)
        self.relu3= nn.ReLU()
        self.conv_incep3_1= nn.Conv2d(channels_list[4],channels_list[5], kernel_size=5, padding=2)
        self.relu4= nn.ReLU()
        self.conv_incep4= nn.MaxPool2d(3, stride=1, padding= 1)
        self.relu5= nn.ReLU()
        self.conv_incep4_1= nn.Conv2d(channels_list[0], channels_list[6], kernel_size=1)
        self.relu6= nn.ReLU()

    def forward(self,x):
        #print('-----------')
        x_1= self.relu(self.conv_incep1(x))
        #print(x_1.size())
        x_2= self.relu(self.conv_incep2_1(self.relu(self.conv_incep2(x))))
        #print(x_2.size())
        x_3= self.relu(self.conv_incep3_1(self.relu(self.conv_incep3(x))))
        #print(x_3.size())
        #x_4= self.conv_incep4(x)
        x_4 = self.relu(self.conv_incep4_1(self.relu(self.conv_incep4(x))))
        #print(x_4.size())
        x_final= torch.cat([x_1,x_2,x_3,x_4],dim=1)
        #print(x_final.size())
        return x_final

class GNet(nn.Module):
    def __init__(self, Incep):
        super(GNet,self).__init__()
        self.in_channels= 64
        self.inception_list= [2,5,2]
        self.conv1= nn.Conv2d(3,64,kernel_size=7, stride=2, padding=3
                              )
        self.relu= nn.ReLU()
        self.mp1= nn.MaxPool2d(3,stride=2, padding=1)
        self.conv2= nn.Conv2d(64,192, kernel_size=3, padding=1)
        self.relu1= nn.ReLU()
        self.conv3= nn.Conv2d(192,192, kernel_size=1)
        self.relu2= nn.ReLU()
        self.mp2= nn.MaxPool2d(3,stride=2, padding=1)
        self.incept_mod_1= self.make_inception_model(Incep, self.inception_list[0])
        self.mp3= nn.MaxPool2d(3,stride=2, padding=1)
        self.incept_mod_2= self.make_inception_model(Incep, self.inception_list[1])

        self.mp4= nn.MaxPool2d(3,stride=2, padding=1)
        self.incept_mod_3= self.make_inception_model(Incep, self.inception_list[2], final='yes')

        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.drop= nn.Dropout(0.4)
        self.linear= nn.Linear(1024,1000)
        

    
    def make_inception_model(self, Incep,no_modules, final=None):
        layers= []
        if no_modules==2 and final==None:
            values={'3a': [192, 64,96,128,16,32,32], 
                    '3b': [256, 128,128,192,32,96,64]}
            for i in values:
                layers.append(Incep(values[i]))
        if no_modules==5:
            values ={'4a': [480,192,96,208,16,48,64],
                     '4b': [512,160,112,224,24,64,64],
                     '4c': [512,128,128,256,24,64,64],
                     '4d': [512,112,144,288,32,64,64],
                     '4e': [528,256,160,320,32,128,128]}
                    
            for i in values:
                layers.append(Incep(values[i]))

        if no_modules==2 and final=='yes':
            values={'5a': [832, 256,160,320,32,128,128], 
                    '5b': [832, 384,192,384,48,128,128]}
            for i in values:
                layers.append(Incep(values[i]))

        return nn.Sequential(*layers)
      
    def forward(self,x):
        x= self.relu(self.conv1(x))
        print(x.size())
        x= self.mp1(x)
        print(x.size())
        x= self.relu(self.conv2(x))
        print(x.size())
        x= self.relu(self.conv3(x))
        print(x.size())
        x= self.mp2(x)
        print(x.size())
        x= self.incept_mod_1(x)
        print(x.size())
        x= self.mp3(x)
        print(x.size())
        x= self.incept_mod_2(x)
        print(x.size())
        x= self.mp4(x)
        print(x.size())
        x= self.incept_mod_3(x)
        print(x.size())
        x= self.avgpool(x)
        print(x.size())
        x=x.reshape(x.shape[0],-1)
        print(x.size())
        x= self.linear(x)
        print(x.size())

        return x

if __name__ == "__main__":
    net= GNet(Incep)
    img= torch.randn(1,3,224,224)
    print(net(img).size())
    print(net)

    '''
    values={'3a': [64,96,128,16,32,32], 
                    '3b': [128,128,192,32,96,64]}
    print(values['3a'])
    for i in values:
        print(i)
    '''