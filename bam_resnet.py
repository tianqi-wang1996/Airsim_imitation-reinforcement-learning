import torch
import torch.nn as nn

# minimum component block for BAM_ResNet34
# flag input--'downsample' to determine whether downsampling will be done in this block 
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        if downsample:
            self.stride = 2
        else:
            self.stride = 1

        self.relu = nn.ReLU()

        self.conv_main = nn.Conv2d(in_channels = self.in_channels,
                                  out_channels = self.out_channels,
                                  kernel_size = 1,
                                  stride = self.stride,
                                  padding = 0,
                                  dilation = 1,
                                  bias = False)
        self.batch_norm_main  = nn.BatchNorm2d(self.out_channels)

        self.conv1 = nn.Conv2d(in_channels = self.in_channels,
                               out_channels = self.out_channels,
                               kernel_size = 3,
                               stride = self.stride,
                               padding = 1,
                               dilation = 1,
                               bias = False)
        self.batch_norm1  = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(in_channels = self.out_channels,
                              out_channels = self.out_channels,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1,
                              dilation = 1,
                              bias = False)
        self.batch_norm2  = nn.BatchNorm2d(self.out_channels) 

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_down = nn.Linear(out_channels, out_channels//16)
        self.bn_down = nn.BatchNorm1d(out_channels//16)
        self.fc_up = nn.Linear(out_channels//16, out_channels)

        self.spatial_conv1 = nn.Conv2d(out_channels, out_channels//16, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.spatial_bn1 = nn.BatchNorm2d(out_channels//16) 
        self.spatial_conv2 = nn.Conv2d(out_channels//16, out_channels//16, kernel_size=3, stride=1, padding=4, dilation=4, bias=False)
        self.spatial_bn2 = nn.BatchNorm2d(out_channels//16) 
        self.spatial_conv3 = nn.Conv2d(out_channels//16, out_channels//16, kernel_size=3, stride=1, padding=4, dilation=4, bias=False)
        self.spatial_bn3 = nn.BatchNorm2d(out_channels//16)
        self.spatial_conv4 = nn.Conv2d(out_channels//16, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.attention_bn = nn.BatchNorm2d(out_channels)
        self.sig = nn.Sigmoid()



    def forward(self, x):
        #main branch
        if self.downsample:
            shortcut = self.conv_main(x)
            shortcut = self.batch_norm_main(shortcut)
        else:
            shortcut = x

        #side branch
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)

        #channel attention
        x1 = self.global_pool(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc_down(x1)
        x1 = self.bn_down(x1)
        x1 = self.relu(x1)
        x1 = self.fc_up(x1)
        x1 = x1.unsqueeze(2).unsqueeze(3).expand_as(x)

        #spatial attention
        x2 = self.spatial_conv1(x)
        x2 = self.spatial_bn1(x2)
        x2 = self.relu(x2)
        x2 = self.spatial_conv2(x2)
        x2 = self.spatial_bn2(x2)
        x2 = self.relu(x2)
        x2 = self.spatial_conv3(x2)
        x2 = self.spatial_bn3(x2)
        x2 = self.relu(x2)        
        x2 = self.spatial_conv4(x2)
        x2 = x2.expand_as(x)

        # if we want to only do channel or spatial attention, just simply change this line 
        attention = x1 + x2
        
        attention = self.attention_bn(attention)
        attention = self.sig(attention)
        
        out = attention * x + shortcut
        out = self.relu(out)

        return out



class BAM_ResNet34(nn.Module):
    def __init__(self):
        super().__init__()

        # remark: In the original ResNet paper, the first convolution layer takes stride=2 to downsample the input
        # ,after this convolution layer there is also another maxpooling layer to do downsample as well.
        # But that was the structure for ImageNet dataset which is 224x224 size. For CIFAR100 dataset, if we keep using
        # this, there will be too many 1x1 dimension in the later layer's output which I did the experiment and found that 
        # the performance is poor. So here I refer to the official CBAM code, do not do downsampling in the first conv layer
        # and delete the maxpooling layer as well.
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.b21 = BottleNeck(64,64)
        self.b22 = BottleNeck(64,64)
        self.b23 = BottleNeck(64,64)

        self.b31 = BottleNeck(64,128,downsample=True)
        self.b32 = BottleNeck(128,128)
        self.b33 = BottleNeck(128,128)      
        self.b34 = BottleNeck(128,128)

        self.b41 = BottleNeck(128,256,downsample=True)
        self.b42 = BottleNeck(256,256)
        self.b43 = BottleNeck(256,256)      
        self.b44 = BottleNeck(256,256)  
        self.b45 = BottleNeck(256,256)
        self.b46 = BottleNeck(256,256)

        self.b51 = BottleNeck(256,512,downsample=True)
        self.b52 = BottleNeck(512,512)
        self.b53 = BottleNeck(512,512)              

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(513, 250)
        self.fc1_bn = nn.BatchNorm1d(250)
        self.fc2 = nn.Linear(250, 100)
        self.fc2_bn = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 3)   

    def forward(self, x, x2):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)

        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)

        x = self.b41(x)
        x = self.b42(x)
        x = self.b43(x)
        x = self.b44(x)
        x = self.b45(x)
        x = self.b46(x)

        x = self.b51(x)
        x = self.b52(x)
        x = self.b53(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x,x2), dim = 1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x) 
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

# minimum component block for BAM_ResNet50
# flag input--'downsample' to determine whether downsampling will be done in this block 
class BottleNeck_50(nn.Module):
    def __init__(self, in_channels, out_channels, first_block=False, downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.first_block = first_block

        if downsample:
            self.stride = 2
        else:
            self.stride = 1

        self.relu = nn.ReLU()

        self.conv_main = nn.Conv2d(in_channels = self.in_channels,
                                  out_channels = self.out_channels,
                                  kernel_size = 1,
                                  stride = self.stride,
                                  padding = 0,
                                  dilation = 1,
                                  bias = False)
        self.batch_norm_main = nn.BatchNorm2d(self.out_channels)


        self.conv1 = nn.Conv2d(in_channels = self.in_channels,
                               out_channels = self.out_channels//4,
                               kernel_size = 1,
                               stride = self.stride,
                               padding = 0,
                               dilation = 1,
                               bias = False)
        self.batch_norm1  = nn.BatchNorm2d(self.out_channels//4)
        self.conv2 = nn.Conv2d(in_channels = self.out_channels//4,
                               out_channels = self.out_channels//4,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               dilation = 1,
                               bias = False)  
        self.batch_norm2  = nn.BatchNorm2d(self.out_channels//4)                           
        self.conv3 = nn.Conv2d(in_channels = self.out_channels//4,
                              out_channels = self.out_channels,
                              kernel_size = 1,
                              stride = 1,
                              padding = 0,
                              dilation = 1,
                              bias = False)
        self.batch_norm3  = nn.BatchNorm2d(self.out_channels)   

              
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_down = nn.Linear(out_channels, out_channels//16)
        self.bn_down = nn.BatchNorm1d(out_channels//16)
        self.fc_up = nn.Linear(out_channels//16, out_channels)

        self.spatial_conv1 = nn.Conv2d(out_channels, out_channels//16, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.spatial_bn1 = nn.BatchNorm2d(out_channels//16) 
        self.spatial_conv2 = nn.Conv2d(out_channels//16, out_channels//16, kernel_size=3, stride=1, padding=4, dilation=4, bias=False)
        self.spatial_bn2 = nn.BatchNorm2d(out_channels//16) 
        self.spatial_conv3 = nn.Conv2d(out_channels//16, out_channels//16, kernel_size=3, stride=1, padding=4, dilation=4, bias=False)
        self.spatial_bn3 = nn.BatchNorm2d(out_channels//16)
        self.spatial_conv4 = nn.Conv2d(out_channels//16, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.attention_bn = nn.BatchNorm2d(out_channels)
        self.sig = nn.Sigmoid()



    def forward(self, x):
        #main branch
        if self.downsample:
            shortcut = self.conv_main(x)
            shortcut = self.batch_norm_main(shortcut)
        elif self.first_block:
            shortcut = self.conv_main(x)
            shortcut = self.batch_norm_main(shortcut)
        else:
            shortcut = x

        #side branch
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)

        #channel attention
        x1 = self.global_pool(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc_down(x1)
        x1 = self.bn_down(x1)
        x1 = self.relu(x1)
        x1 = self.fc_up(x1)
        x1 = x1.unsqueeze(2).unsqueeze(3).expand_as(x)

        #spatial attention
        x2 = self.spatial_conv1(x)
        x2 = self.spatial_bn1(x2)
        x2 = self.relu(x2)
        x2 = self.spatial_conv2(x2)
        x2 = self.spatial_bn2(x2)
        x2 = self.relu(x2)
        x2 = self.spatial_conv3(x2)
        x2 = self.spatial_bn3(x2)
        x2 = self.relu(x2)        
        x2 = self.spatial_conv4(x2)
        x2 = x2.expand_as(x)

        # if we want to only do channel or spatial attention, just simply change this line 
        attention = x1 + x2

        attention = self.attention_bn(attention)
        attention = self.sig(attention)

        out = attention * x + shortcut
        out = self.relu(out)

        return out


class BAM_ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        # remark: In the original ResNet paper, the first convolution layer takes stride=2 to downsample the input
        # ,after this convolution layer there is also another maxpooling layer to do downsample as well.
        # But that was the structure for ImageNet dataset which is 224x224 size. For CIFAR100 dataset, if we keep using
        # this, there will be too many 1x1 dimension in the later layer's output which I did the experiment and found that 
        # the performance is poor. So here I refer to the official CBAM code, do not do downsampling in the first conv layer
        # and delete the maxpooling layer as well.
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.b21 = BottleNeck_50(64,256,first_block=True)
        self.b22 = BottleNeck_50(256,256)
        self.b23 = BottleNeck_50(256,256)

        self.b31 = BottleNeck_50(256,512,downsample=True)
        self.b32 = BottleNeck_50(512,512)
        self.b33 = BottleNeck_50(512,512)      
        self.b34 = BottleNeck_50(512,512)

        self.b41 = BottleNeck_50(512,1024,downsample=True)
        self.b42 = BottleNeck_50(1024,1024)
        self.b43 = BottleNeck_50(1024,1024)      
        self.b44 = BottleNeck_50(1024,1024)        
        self.b45 = BottleNeck_50(1024,1024)      
        self.b46 = BottleNeck_50(1024,1024)  

        self.b51 = BottleNeck_50(1024,2048,downsample=True)
        self.b52 = BottleNeck_50(2048,2048)
        self.b53 = BottleNeck_50(2048,2048)                 

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2049, 250)
        self.fc2 = nn.Linear(250, 100)
        self.fc3 = nn.Linear(100, 3)  


    def forward(self, x, x2):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)

        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)

        x = self.b41(x)
        x = self.b42(x)
        x = self.b43(x)
        x = self.b44(x)
        x = self.b45(x)
        x = self.b46(x)

        x = self.b51(x)
        x = self.b52(x)
        x = self.b53(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x,x2), dim = 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x