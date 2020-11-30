import torch
import torch.nn as nn

DILATION= [[1,1,1,1,1,1],
            [1,1,2,2,2],
            [2,2,4,4,4]]



class block(nn.Module):
    """
    This Class will define layers in each block

    """
    def __init__(self, in_channels, out_channels,type_int, padding_last=0,
    identity_downsample=None, stride=1):

        super(block,self).__init__()
        self.expansion= 4
        #print(type_int)
        self.conv1= nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=1, padding=0, dilation=DILATION[type_int][0])
        self.bn1= nn.BatchNorm2d(out_channels)
        self.conv2= nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1,dilation=DILATION[type_int][1])
        self.bn2= nn.BatchNorm2d(out_channels)
        self.conv3= nn.Conv2d(out_channels,out_channels*self.expansion,
                             kernel_size=1, stride=1, padding=padding_last,dilation=DILATION[type_int][2])
        self.bn3= nn.BatchNorm2d(out_channels*self.expansion)
        self.relu= nn.ReLU()
        self.identity_downsample= identity_downsample
        #self.type_int= type_int

    def forward(self, x):
        #print('-----------')
        identity=x
        x= self.conv1(x)
        #print(x.size())
        x= self.bn1(x)
        x= self.relu(x)
        x= self.conv2(x)
        #print(x.size())
        x= self.bn2(x)
        x= self.relu(x)
        x= self.conv3(x)
        #print('Prev',x.size())  
        x= self.bn3(x)
        #print('Id',identity.size())
        #print('identity_downsample',self.identity_downsample)
        
        # identity downsample that residual mapping
        if self.identity_downsample is not None:
            identity=self.identity_downsample(identity) #seems like neural networ

        #print('Id1',identity.size())        
        x+= identity
        #print('identity_downsample', identity.size())
        x= self.relu(x)
        #print('-----------')
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        """
        Initial Layers
        """
        self.in_channels= 64
        self.conv1= nn.Conv2d(image_channels,64, kernel_size=7, stride=2, padding=3)
        self.bn1= nn.BatchNorm2d(64)
        self.relu= nn.ReLU()
        self.maxpool= nn.MaxPool2d(kernel_size=3,stride=2, padding=1)

        #Resnet Layers
        self.layer1= self._make_layer(block, layers[0], out_channels=64, stride=1,type_int=0) #3
        self.layer2= self._make_layer(block, layers[1], out_channels=128, stride=2,type_int=0)
        self.layer3= self._make_layer(block, layers[2], out_channels=256, stride=1,type_int=1)
        self.layer4= self._make_layer(block, layers[3], out_channels=512, stride=1,type_int=2,padding_last=1) 

        self.pool1= nn.AdaptiveAvgPool2d((2,2))


    def forward(self,x):
        x= self.conv1(x)
        x= self.bn1(x)
        x= self.relu(x)
        x= self.maxpool(x)

        x= self.layer1(x)
        print(x.size())
        x= self.layer2(x)
        print(x.size())
        #print('going in layer 3')
        x= self.layer3(x)
        print(x.size())
        #print('going in layer 4')
        x= self.layer4(x)
        print(x.size())
        x= self.pool1(x)
        print(x.size())


        return x

    def _make_layer(self, block, num_residual_blocks,out_channels, stride, type_int, padding_last=0):
        identity_downsample= None
        layers=[]

        if stride!= 1 or self.in_channels!=out_channels*4:
            identity_downsample= nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, 
                                    kernel_size=1, stride=stride),
                                               nn.BatchNorm2d(out_channels*4))
                    
        layers.append(block(self.in_channels, out_channels,type_int,padding_last,identity_downsample, stride))
        self.in_channels= out_channels*4 #256
        
        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels, out_channels,type_int,padding_last)) #256-> 64, 64*4

        return nn.Sequential(*layers)

def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3,4,6,3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(block, [3,4,23,3], img_channels, num_classes)


if __name__ == "__main__":
    x= torch.randn(2,3,224,224)
    net= ResNet50()
    y= net(x)
    print(y.shape)
    #print(net)



    

