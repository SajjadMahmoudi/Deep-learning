class Fuse(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    

    
    def __init__(self, in_channels=IN_CHANNELS, out_channels=N_CLASSES):
        super(Fuse, self).__init__()
        self.height = height = 256
        self.width = width = 256
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        
        #IRGB seg
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
        
        #OUT
        self.conv2_2_OUT_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_OUT_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_OUT_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_OUT_D_bn = nn.BatchNorm2d(64)
        
        self.conv1_2_OUT_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_OUT_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_OUT_D = nn.Conv2d(64, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        
    def forward(self, rgb_input, depth_input):
        # Encoder block 1_IRGB
        x = self.conv1_1_bn(F.relu(self.conv1_1(rgb_input)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)
        
        # Encoder block 2_IRGB
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)
        
        # Encoder block 3_IRGB
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)
        
        # Encoder block 4_IRGB
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)
        
        # Encoder block 5_IRGB
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)
        
        # Decoder block 5_IRGB
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x_5 = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4_IRGB
        x = self.unpool(x_5, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x_4 = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3_IRGB
        x = self.unpool(x_4, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x_3 = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2_IRGB
        x = self.unpool(x_3, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x_2 = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1_IRGB
        x = self.unpool(x_2, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x_1 = self.conv1_1_D(x)
        
        
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
        
        #fusion

        z_1 = torch.add(x_1, y_1)

        cat = torch.cat([x_2, y_2], dim=1)
        
        # Decoder block 2_OUT
        z = self.conv2_2_OUT_D_bn(F.relu(self.conv2_2_OUT_D(cat)))
        z_2 = self.conv2_1_OUT_D_bn(F.relu(self.conv2_1_OUT_D(z)))
        # Decoder block 1_OUT
        z = self.unpool(z_2, mask1_D)
        z = self.conv1_2_OUT_D_bn(F.relu(self.conv1_2_OUT_D(z)))
        z_3 = self.conv1_1_OUT_D(z)
        
        z_4 = torch.add(z_1, z_3)
        
        output = F.log_softmax(z_4)
        return output
