import torch
import torch.nn as nn

class depthwise_pointwise(nn.Module):
    def __init__(self, channels, stride=1):
        super(depthwise_pointwise, self).__init__()
        self.depthwise_layer= nn.Sequential(
            nn.Conv2d(channels[0], channels[0],3,
                      groups=channels[0], padding=1, stride=stride),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0],channels[1],1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
        )
    
    def forward(self,x):
        return self.depthwise_layer(x)

class Mobilenet(nn.Module):
    def __init__(self,depthwise_pointwise):
        super(Mobilenet, self).__init__()
        self.conv1= nn.Conv2d(3,32, 3, stride=2, padding=1)
        self.bn1= nn.BatchNorm2d(32)
        self.relu= nn.ReLU()
        #depthwise sep
        self.dw_s1= depthwise_pointwise([32,64], stride=1)
        self.dw_s2= depthwise_pointwise([64,128], stride=2)
        self.dw_s3= depthwise_pointwise([128,128])
        self.dw_s4= depthwise_pointwise([128,256], stride=2)
        self.dw_s5= depthwise_pointwise([256,256])
        self.dw_s6= depthwise_pointwise([256,512], stride=2)
        # x5 layer stack
        self.dw_x5= depthwise_pointwise([512,512])
        # dw layer
        self.dw_s7= depthwise_pointwise([512,1024], stride=2)
        self.dw_s8= depthwise_pointwise([1024,1024])
        #avg pool
        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.fc= nn.Linear(1024,1000)


    def forward(self,x):
        x= self.relu(self.bn1(self.conv1(x)))
        print(x.size())
        x= self.dw_s1(x)
        print(x.size())
        x= self.dw_s2(x)
        print(x.size())
        x= self.dw_s3(x)
        print(x.size())
        x= self.dw_s4(x)
        print(x.size())
        x= self.dw_s5(x)
        print(x.size())
        x= self.dw_s6(x)
        print(x.size())
        for i in range(5):
            x= self.dw_x5(x)
        print(x.size())
        x= self.dw_s7(x)
        print(x.size())
        x= self.dw_s8(x)
        print(x.size())
        x= self.avgpool(x)
        print(x.size())
        x=x.view(x.shape[0],-1)
        x= self.fc(x)
        print(x.size())

        return x
        
if __name__ == "__main__":
    mb= Mobilenet(depthwise_pointwise)
    img= torch.randn(1,3,224,224)
    mb(img)
    print(mb)