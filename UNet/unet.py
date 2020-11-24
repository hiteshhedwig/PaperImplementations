import torch 
import torch.nn as nn
import copy
import argparse

parser= argparse.ArgumentParser(description='UNet segmentation')
parser.add_argument('--debug', type=bool, default=False,
                    choices=[True, False])
parser.add_argument('--network', type=bool, default=False,
                    choices=[True, False])

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.in_channels= 1

        #downsample_4
        self.downsample1= self.make_inlayers(self.in_channels, 64)
        self.maxpool1= nn.MaxPool2d(2,stride=2)
        self.downsample2= self.make_inlayers(64, 128)
        self.maxpool2= nn.MaxPool2d(2,stride=2)
        self.downsample3= self.make_inlayers(128, 256)
        self.maxpool3= nn.MaxPool2d(2,stride=2)
        self.downsample4= self.make_inlayers(256, 512)
        self.maxpool4= nn.MaxPool2d(2,stride=2)
        self.downsample5= self.make_inlayers(512, 1024)

        #upside upsample
        self.upsample6=nn.ConvTranspose2d(1024,512,kernel_size=2, stride=2)
        self.u7= self.make_inlayers(1024,512)

        ##2
        self.upsample8=nn.ConvTranspose2d(512,256,kernel_size=2, stride=2)
        self.u9= self.make_inlayers(512,256)

        ##3
        self.upsample10=nn.ConvTranspose2d(256,128,kernel_size=2, stride=2)
        self.u11=self.make_inlayers(256,128)

        ##4
        self.upsample12= nn.ConvTranspose2d(128,64,kernel_size=2, stride=2)
        self.u13= self.make_inlayers(128,64)

        ##final
        self.final=nn.Conv2d(64,2,kernel_size=1, stride=1)





    def make_inlayers(self, in_channels, out_channels, k=2):
        layers=[]
        if k==2:
            return nn.Sequential(
                            nn.Conv2d(in_channels, out_channels,3),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels,3),
                            nn.ReLU(),
            )

    def cropped_layer(self,target,image):
        target_size=target.size()[2]
        image_size=image.size()[2]
        delta= image_size- target_size
        delta= delta//2
        return image[:,:,delta:image_size-delta,delta:image_size-delta]       

            
    def forward(self,x,debug= False):
        #debug= self.debug
        print(debug)
        save_layer=[]
        x= self.downsample1(x)
        save_layer.append(x)
        if debug: print(x.size())
        x= self.maxpool1(x)
        if debug: print(x.size())
        x= self.downsample2(x)
        save_layer.append(x)
        if debug: print(x.size())
        x= self.maxpool2(x)
        x= self.downsample3(x)
        save_layer.append(x)
        if debug: print(x.size())
        x= self.maxpool3(x)
        if debug: print(x.size())
        x= self.downsample4(x)
        save_layer.append(x)
        if debug: print(x.size())
        x= self.maxpool4(x)
        if debug: print(x.size())
        x= self.downsample5(x)
        if debug: print(x.size())

        x=self.upsample6(x)
        if debug: print(x.size())
        conc_layer= self.cropped_layer(x,save_layer[3])
        #print(conc_layer.size())
        x=torch.cat([x,conc_layer], dim=1)
        if debug: print(x.size())
        x=self.u7(x)
        if debug: print(x.size())

        ##2
        x=self.upsample8(x)
        if debug: print(x.size())
        conc_layer= self.cropped_layer(x,save_layer[2])
        #print(conc_layer.size())
        x=torch.cat([x,conc_layer], dim=1)
        if debug: print(x.size())
        x=self.u9(x)
        if debug: print(x.size())

        ##3
        x=self.upsample10(x)
        if debug: print(x.size())
        conc_layer= self.cropped_layer(x,save_layer[1])
        #print(conc_layer.size())
        x=torch.cat([x,conc_layer], dim=1)
        if debug: print(x.size())
        x=self.u11(x)
        if debug: print(x.size())

        ##4
        x=self.upsample12(x)
        if debug: print(x.size())
        conc_layer= self.cropped_layer(x,save_layer[0])
        #print(conc_layer.size())
        x=torch.cat([x,conc_layer], dim=1)
        if debug: print(x.size())
        x=self.u13(x)
        if debug: print(x.size())

        ##final
        x= self.final(x)
        if debug: print(x.size())
    
        

if __name__ == "__main__":
    x= torch.randn(1,1,572,572)
    args= parser.parse_args()
    net=UNet()
    print(args.debug)
    net(x,args.debug)
    if args.network:
        print('>     Model')
        print(net)
    