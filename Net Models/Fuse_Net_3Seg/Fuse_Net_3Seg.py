class FuseNet_3seg(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    

    
    def __init__(self, in_channels=IN_CHANNELS, out_channels=N_CLASSES):
        super(FuseNet_3, self).__init__()
        self.height = height = 256
        self.width = width = 256
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        
        #IRRGB seg
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)
        
        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)
        
        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)
        
        self.dropout_RGB = nn.Dropout2d(p=0.1)
        
        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)
        
        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)
        
        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)
        #____________________________________________________________________________________________#
        #IRRG seg
        self.conv1_1_IRRG = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_1_IRRG_bn = nn.BatchNorm2d(64)
        self.conv1_2_IRRG = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_IRRG_bn = nn.BatchNorm2d(64)
        
        self.conv2_1_IRRG = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_IRRG_bn = nn.BatchNorm2d(128)
        self.conv2_2_IRRG = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_IRRG_bn = nn.BatchNorm2d(128)
        
        self.conv3_1_IRRG = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_IRRG_bn = nn.BatchNorm2d(256)
        self.conv3_2_IRRG = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_IRRG_bn = nn.BatchNorm2d(256)
        self.conv3_3_IRRG = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_IRRG_bn = nn.BatchNorm2d(256)
        
        self.conv4_1_IRRG = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_IRRG_bn = nn.BatchNorm2d(512)
        self.conv4_2_IRRG = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_IRRG_bn = nn.BatchNorm2d(512)
        self.conv4_3_IRRG = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_IRRG_bn = nn.BatchNorm2d(512)
        
        self.conv5_1_IRRG = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_IRRG_bn = nn.BatchNorm2d(512)
        self.conv5_2_IRRG = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_IRRG_bn = nn.BatchNorm2d(512)
        self.conv5_3_IRRG = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_IRRG_bn = nn.BatchNorm2d(512)
        
        self.conv5_3_D_IRRG = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_IRRG_bn = nn.BatchNorm2d(512)
        self.conv5_2_D_IRRG = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_IRRG_bn = nn.BatchNorm2d(512)
        self.conv5_1_D_IRRG = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_IRRG_bn = nn.BatchNorm2d(512)
        
        self.conv4_3_D_IRRG = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_IRRG_bn = nn.BatchNorm2d(512)
        self.conv4_2_D_IRRG = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_IRRG_bn = nn.BatchNorm2d(512)
        self.conv4_1_D_IRRG = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_IRRG_bn = nn.BatchNorm2d(256)
        
        self.dropout_IRRG = nn.Dropout2d(p=0.1)
        
        self.conv3_3_D_IRRG = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_IRRG_bn = nn.BatchNorm2d(256)
        self.conv3_2_D_IRRG = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_IRRG_bn = nn.BatchNorm2d(256)
        self.conv3_1_D_IRRG = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_IRRG_bn = nn.BatchNorm2d(128)
        
        self.conv2_2_D_IRRG = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_IRRG_bn = nn.BatchNorm2d(128)
        self.conv2_1_D_IRRG = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_IRRG_bn = nn.BatchNorm2d(64)
        
        self.conv1_2_D_IRRG = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_IRRG_bn = nn.BatchNorm2d(64)
        self.conv1_1_D_IRRG = nn.Conv2d(64, out_channels, 3, padding=1)
        #____________________________________________________________________________________________#
        #depth seg
        self.conv1_1_Depth = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_1_Depth_bn = nn.BatchNorm2d(64)
        self.conv1_2_Depth = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_Depth_bn = nn.BatchNorm2d(64)
        
        self.conv2_1_Depth = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_Depth_bn = nn.BatchNorm2d(128)
        self.conv2_2_Depth = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_Depth_bn = nn.BatchNorm2d(128)
        
        self.conv3_1_Depth = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_Depth_bn = nn.BatchNorm2d(256)
        self.conv3_2_Depth = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_Depth_bn = nn.BatchNorm2d(256)
        self.conv3_3_Depth = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_Depth_bn = nn.BatchNorm2d(256)
        
        self.conv4_1_Depth = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_Depth_bn = nn.BatchNorm2d(512)
        self.conv4_2_Depth = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_Depth_bn = nn.BatchNorm2d(512)
        self.conv4_3_Depth = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_Depth_bn = nn.BatchNorm2d(512)
        
        self.conv5_1_Depth = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_Depth_bn = nn.BatchNorm2d(512)
        self.conv5_2_Depth = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_Depth_bn = nn.BatchNorm2d(512)
        self.conv5_3_Depth = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_Depth_bn = nn.BatchNorm2d(512)
        
        self.conv5_3_Depth_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_Depth_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_Depth_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_Depth_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_Depth_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_Depth_D_bn = nn.BatchNorm2d(512)
        
        self.conv4_3_Depth_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_Depth_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_Depth_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_Depth_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_Depth_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_Depth_D_bn = nn.BatchNorm2d(256)
        
        self.dropout_Depth = nn.Dropout2d(p=0.1)
        
        self.conv3_3_Depth_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_Depth_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_Depth_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_Depth_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_Depth_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_Depth_D_bn = nn.BatchNorm2d(128)
        
        self.conv2_2_Depth_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_Depth_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_Depth_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_Depth_D_bn = nn.BatchNorm2d(64)
        
        self.conv1_2_Depth_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_Depth_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_Depth_D = nn.Conv2d(64, out_channels, 3, padding=1)
        #____________________________________________________________________________________________#

        #OUT
        self.conv3_3_OUT_D = nn.Conv2d(192, 192, 3, padding=1)
        self.conv3_3_OUT_D_bn = nn.BatchNorm2d(192)
        self.conv3_2_OUT_D = nn.Conv2d(192, 192, 3, padding=1)
        self.conv3_2_OUT_D_bn = nn.BatchNorm2d(192)
        self.conv3_1_OUT_D = nn.Conv2d(192, 128, 3, padding=1)
        self.conv3_1_OUT_D_bn = nn.BatchNorm2d(128)
        
        self.dropout_Out = nn.Dropout2d(p=0.5)
        
        self.conv2_2_OUT_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_OUT_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_OUT_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_OUT_D_bn = nn.BatchNorm2d(64)
        
        self.conv1_2_OUT_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_OUT_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_OUT_D = nn.Conv2d(64, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        
    def forward(self, rgb_input,irrg_input, depth_input):
        # Encoder block 1_RGB
        
        x = self.conv1_1_bn(F.relu(self.conv1_1(rgb_input)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)
        
        # Encoder block 2_RGB
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)
        
        # Encoder block 3_RGB
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)
        
        # Encoder block 4_RGB
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)
        
        # Encoder block 5_RGB
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)
        
        # Decoder block 5_RGB
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x_5 = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4_RGB
        x = self.unpool(x_5, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x_4 = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        x_4 = self.dropout_RGB(x_4)
        # Decoder block 3_RGB
        x = self.unpool(x_4, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x_3 = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2_RGB
        x = self.unpool(x_3, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x_2 = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1_RGB
        x = self.unpool(x_2, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x_1 = self.conv1_1_D(x)
        x_1 = F.log_softmax(x_1,dim=1)
        #______________________________________________________________________________#
        # Encoder block 1_IRRG
        R = self.conv1_1_IRRG_bn(F.relu(self.conv1_1_IRRG(irrg_input)))
        R = self.conv1_2_IRRG_bn(F.relu(self.conv1_2_IRRG(R)))
        R, mask1_R = self.pool(R)
        
        # Encoder block 2_IRRG
        R = self.conv2_1_IRRG_bn(F.relu(self.conv2_1_IRRG(R)))
        R = self.conv2_2_IRRG_bn(F.relu(self.conv2_2_IRRG(R)))
        R, mask2_R = self.pool(R)
        
        # Encoder block 3_IRRG
        R = self.conv3_1_IRRG_bn(F.relu(self.conv3_1_IRRG(R)))
        R = self.conv3_2_IRRG_bn(F.relu(self.conv3_2_IRRG(R)))
        R = self.conv3_3_IRRG_bn(F.relu(self.conv3_3_IRRG(R)))
        R, mask3_R = self.pool(R)
        
        # Encoder block 4_IRRG
        R = self.conv4_1_IRRG_bn(F.relu(self.conv4_1_IRRG(R)))
        R = self.conv4_2_IRRG_bn(F.relu(self.conv4_2_IRRG(R)))
        R = self.conv4_3_IRRG_bn(F.relu(self.conv4_3_IRRG(R)))
        R, mask4_R = self.pool(R)
        
        # Encoder block 5_IRRG
        R = self.conv5_1_IRRG_bn(F.relu(self.conv5_1_IRRG(R)))
        R = self.conv5_2_IRRG_bn(F.relu(self.conv5_2_IRRG(R)))
        R = self.conv5_3_IRRG_bn(F.relu(self.conv5_3_IRRG(R)))
        R, mask5_R = self.pool(R)
        
        # Decoder block 5_IRRG
        R = self.unpool(R, mask5_R)
        R = self.conv5_3_D_IRRG_bn(F.relu(self.conv5_3_D_IRRG(R)))
        R = self.conv5_2_D_IRRG_bn(F.relu(self.conv5_2_D_IRRG(R)))
        R_5 = self.conv5_1_D_IRRG_bn(F.relu(self.conv5_1_D_IRRG(R)))
        
        # Decoder block 4_IRRG
        R = self.unpool(R_5, mask4_R)
        R = self.conv4_3_D_IRRG_bn(F.relu(self.conv4_3_D_IRRG(R)))
        R = self.conv4_2_D_IRRG_bn(F.relu(self.conv4_2_D_IRRG(R)))
        R_4 = self.conv4_1_D_IRRG_bn(F.relu(self.conv4_1_D_IRRG(R)))
        
        R_4 = self.dropout_IRRG(R_4)
        # Decoder block 3_IRRG
        R = self.unpool(R_4, mask3_R)
        R = self.conv3_3_D_IRRG_bn(F.relu(self.conv3_3_D_IRRG(R)))
        R = self.conv3_2_D_IRRG_bn(F.relu(self.conv3_2_D_IRRG(R)))
        R_3 = self.conv3_1_D_IRRG_bn(F.relu(self.conv3_1_D_IRRG(R)))
        
        # Decoder block 2_IRRG
        R = self.unpool(R_3, mask2_R)
        R = self.conv2_2_D_IRRG_bn(F.relu(self.conv2_2_D_IRRG(R)))
        R_2 = self.conv2_1_D_IRRG_bn(F.relu(self.conv2_1_D_IRRG(R)))
        
        # Decoder block 1_IRRG
        R = self.unpool(R_2, mask1_R)
        R = self.conv1_2_D_IRRG_bn(F.relu(self.conv1_2_D_IRRG(R)))
        R_1 = self.conv1_1_D_IRRG(R)
        R_1 = F.log_softmax(R_1,dim=1)
        
        #______________________________________________________________________________#
        
        # Encoder block 1_Depth
        y = self.conv1_1_Depth_bn(F.relu(self.conv1_1_Depth(depth_input)))
        y = self.conv1_2_Depth_bn(F.relu(self.conv1_2_Depth(y)))
        y, mask1_D = self.pool(y)
        
        # Encoder block 2_Depth
        y = self.conv2_1_Depth_bn(F.relu(self.conv2_1_Depth(y)))
        y = self.conv2_2_Depth_bn(F.relu(self.conv2_2_Depth(y)))
        y, mask2_D = self.pool(y)
        
        # Encoder block 3_Depth
        y = self.conv3_1_Depth_bn(F.relu(self.conv3_1_Depth(y)))
        y = self.conv3_2_Depth_bn(F.relu(self.conv3_2_Depth(y)))
        y = self.conv3_3_Depth_bn(F.relu(self.conv3_3_Depth(y)))
        y, mask3_D = self.pool(y)
        
        # Encoder block 4_Depth
        y = self.conv4_1_Depth_bn(F.relu(self.conv4_1_Depth(y)))
        y = self.conv4_2_Depth_bn(F.relu(self.conv4_2_Depth(y)))
        y = self.conv4_3_Depth_bn(F.relu(self.conv4_3_Depth(y)))
        y, mask4_D = self.pool(y)
        
        # Encoder block 5_Depth
        y = self.conv5_1_Depth_bn(F.relu(self.conv5_1_Depth(y)))
        y = self.conv5_2_Depth_bn(F.relu(self.conv5_2_Depth(y)))
        y = self.conv5_3_Depth_bn(F.relu(self.conv5_3_Depth(y)))
        y, mask5_D = self.pool(y)
        
        # Decoder block 5_Depth
        y = self.unpool(y, mask5_D)
        y = self.conv5_3_Depth_D_bn(F.relu(self.conv5_3_Depth_D(y)))
        y = self.conv5_2_Depth_D_bn(F.relu(self.conv5_2_Depth_D(y)))
        y_5 = self.conv5_1_Depth_D_bn(F.relu(self.conv5_1_Depth_D(y)))
        
        # Decoder block 4_Depth
        y = self.unpool(y_5, mask4_D)
        y = self.conv4_3_Depth_D_bn(F.relu(self.conv4_3_Depth_D(y)))
        y = self.conv4_2_Depth_D_bn(F.relu(self.conv4_2_Depth_D(y)))
        y_4 = self.conv4_1_Depth_D_bn(F.relu(self.conv4_1_Depth_D(y)))
        
        y_4 = self.dropout_Depth(y_4)
        # Decoder block 3_Depth
        y = self.unpool(y_4, mask3_D)
        y = self.conv3_3_Depth_D_bn(F.relu(self.conv3_3_Depth_D(y)))
        y = self.conv3_2_Depth_D_bn(F.relu(self.conv3_2_Depth_D(y)))
        y_3 = self.conv3_1_Depth_D_bn(F.relu(self.conv3_1_Depth_D(y)))
        
        # Decoder block 2_Depth
        y = self.unpool(y_3, mask2_D)
        y = self.conv2_2_Depth_D_bn(F.relu(self.conv2_2_Depth_D(y)))
        y_2 = self.conv2_1_Depth_D_bn(F.relu(self.conv2_1_Depth_D(y)))
        
        # Decoder block 1_Depth
        y = self.unpool(y_2, mask1_D)
        y = self.conv1_2_Depth_D_bn(F.relu(self.conv1_2_Depth_D(y)))
        y_1 = self.conv1_1_Depth_D(y)
        y_1 = F.log_softmax(y_1,dim=1)
        #________________________________________________________________________#
        #fusion

        ADD_1 = torch.add(x_1, y_1)
        ADD_2 = torch.add(ADD_1, R_1)

        cat_1 = torch.cat([x_2, y_2], dim=1)
        cat_2 = torch.cat([cat_1, R_2], dim=1)
        
        # Decoder block 3_OUT
        z = self.conv3_3_OUT_D_bn(F.relu(self.conv3_3_OUT_D(cat_2)))
        z = self.conv3_2_OUT_D_bn(F.relu(self.conv3_2_OUT_D(z)))
        z_1 = self.conv3_1_OUT_D_bn(F.relu(self.conv3_1_OUT_D(z)))
        
        z_1 = self.dropout_Out(z_1)
        # Decoder block 2_OUT
        z = self.conv2_2_OUT_D_bn(F.relu(self.conv2_2_OUT_D(z_1)))
        z_2 = self.conv2_1_OUT_D_bn(F.relu(self.conv2_1_OUT_D(z)))
        # Decoder block 1_OUT
        z = self.unpool(z_2, mask1_R)
        z = self.conv1_2_OUT_D_bn(F.relu(self.conv1_2_OUT_D(z)))
        z_3 = self.conv1_1_OUT_D(z)
        z_3 = F.log_softmax(z_3,dim=1)
        
        ADD_3 = torch.add(ADD_2, z_3)
        
        output = F.log_softmax(ADD_3,dim=1)
        return output
