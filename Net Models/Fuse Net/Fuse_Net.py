import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class FuseNet(nn.Module):
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, num_labels=6, gpu_device=0, use_class=True):
        super(FuseNet, self).__init__()

        

        bn_moment = 0.1
        

        
        # DEPTH ENCODER
        self.CBR1_1D = nn.Sequential(
            nn.Conv2d(1,64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR1_2D = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device), 
        )
                
        self.CBR2_1D = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),         
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR2_2D = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),         
            nn.ReLU().cuda(gpu_device),
        )
        
        self.CBR3_1D = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR3_2D = nn.Sequential(
            nn.Conv2d(256,256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR3_3D = nn.Sequential(
            nn.Conv2d(256,256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        
        #self.dropout3_d = nn.Dropout(p=0.5).cuda(gpu_device)
        
        
        self.CBR4_1D = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR4_2D = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR4_3D = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        
        #self.dropout4_d = nn.Dropout(p=0.5).cuda(gpu_device)

        
        self.CBR5_1D = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR5_2D = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR5_3D = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )

        
        # RGB ENCODER
        self.CBR1_1RGB = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR1_2RGB = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        
        self.CBR2_1RGB = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR2_2RGB = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )

        self.CBR3_1RGB = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR3_2RGB = nn.Sequential(
            nn.Conv2d(256,256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR3_3RGB = nn.Sequential(
            nn.Conv2d(256,256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )

        #self.dropout3_rgb = nn.Dropout(p=0.5).cuda(gpu_device)
        
        self.CBR4_1RGB = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR4_2RGB = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR4_3RGB = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),            
        )
        
        #self.dropout4_rgb = nn.Dropout(p=0.5).cuda(gpu_device)
        
        self.CBR5_1RGB = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR5_2RGB = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR5_3RGB = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        
        #self.dropout5_rgb = nn.Dropout(p=0.5).cuda(gpu_device)

        
        # RGB DECODER
        self.CBR5_1Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR5_2Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR5_3Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        
        #self.dropout5_Dec = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR4_1Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR4_2Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR4_3Dec = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        
        #self.dropout4_Dec = nn.Dropout(p=0.5).cuda(gpu_device)
        
        self.CBR3_1Dec = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR3_2Dec = nn.Sequential(
            nn.Conv2d(256,  128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        
        #self.dropout3_Dec = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR2_1Dec = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        self.CBR2_2Dec = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )

        self.CBR1_1Dec = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )
        
        self.CBR1_2Dec = nn.Conv2d(64, num_labels, kernel_size=3, padding=1).cuda(gpu_device)
        
        self.initialize_weights()
        print('[INFO] FuseNet model has been created')
        
        
        
    #Initialization for the linear layers in the classification head
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_out = size[0]  # number of rows
                fan_in = size[1]  # number of columns
                variance = np.sqrt(4.0/(fan_in + fan_out))
                m.weight.data.normal_(0.0, variance)
                
    def forward(self, rgb_inputs, depth_inputs):
        # DEPTH ENCODER
        # Stage 1
        #print("depth_inputs IS : ", depth_inputs)
        #print("rgb_inputs IS : ", rgb_inputs)
        #print("depth_inputs shape : ", depth_inputs.shape)
        #print("rgb_inputs shape : ", rgb_inputs.shape)
        
        x = self.CBR1_1D(depth_inputs)
        x_1 = self.CBR1_2D(x)
        x, id1_d = F.max_pool2d(x_1, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x = self.CBR2_1D(x)
        x_2 = self.CBR2_2D(x)
        x, id2_d = F.max_pool2d(x_2, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x = self.CBR3_1D(x)
        x = self.CBR3_2D(x)
        x_3 = self.CBR3_3D(x)
        x, id3_d = F.max_pool2d(x_3, kernel_size=2, stride=2, return_indices=True)
        #x = self.dropout3_d(x)

        # Stage 4
        x = self.CBR4_1D(x)
        x = self.CBR4_2D(x)
        x_4 = self.CBR4_3D(x)
        x, id4_d = F.max_pool2d(x_4, kernel_size=2, stride=2, return_indices=True)
        #x = self.dropout4_d(x)
        
        # Stage 5
        x = self.CBR5_1D(x)
        x = self.CBR5_2D(x)
        x_5 = self.CBR5_3D(x)

        # RGB ENCODER
        # Stage 1
        y = self.CBR1_1RGB(rgb_inputs)
        y = self.CBR1_2RGB(y)
        y = torch.add(y, x_1)
        y, id1 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        y = self.CBR2_1RGB(y)
        y = self.CBR2_2RGB(y)
        y = torch.add(y, x_2)
        y, id2 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        
        # Stage 3
        y = self.CBR3_1RGB(y)
        y = self.CBR3_2RGB(y)
        y = self.CBR3_3RGB(y)
        y = torch.add(y, x_3)
        y, id3 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        #y = self.dropout3_rgb(y)

        # Stage 4
        y = self.CBR4_1RGB(y)
        y = self.CBR4_2RGB(y)
        y = self.CBR4_3RGB(y)
        y = torch.add(y,x_4)
        y, id4 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        #y = self.dropout4_rgb(y)

        # Stage 5
        y = self.CBR5_1RGB(y)
        y = self.CBR5_2RGB(y)
        y = self.CBR5_3RGB(y)
        y = torch.add(y, x_5)
        y_size = y.size()
        y, id5 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        #y = self.dropout5_rgb(y)
        

        # DECODER
        # Stage 5 dec
        y = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)
        y = self.CBR5_1Dec(y)
        y = self.CBR5_2Dec(y)
        y = self.CBR5_3Dec(y)
        #y = self.dropout5_Dec(y)

        # Stage 4 dec
        y = F.max_unpool2d(y, id4, kernel_size=2, stride=2)
        y = self.CBR4_1Dec(y)
        y = self.CBR4_2Dec(y)
        y = self.CBR4_3Dec(y)
        #y = self.dropout4_Dec(y)

        # Stage 3 dec
        y = F.max_unpool2d(y, id3, kernel_size=2, stride=2)
        y = self.CBR3_1Dec(y)
        y = self.CBR3_2Dec(y)
        #y = self.dropout3_Dec(y)

        # Stage 2 dec
        y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
        y = self.CBR2_1Dec(y)
        y = self.CBR2_2Dec(y)

        # Stage 1 dec
        y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
        y = self.CBR1_1Dec(y)
        y = self.CBR1_2Dec(y)
        

        
        return y
    
