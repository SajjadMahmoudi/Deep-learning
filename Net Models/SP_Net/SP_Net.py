class SP_Net(nn.Module):
    
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels=IN_CHANNELS, out_channels=N_CLASSES):
        img_size = 256
        img_size_8 = 32
        n_classes = 6
        block_config = [3, 4, 6, 3]  # resnet50
        
        super(SP_Net, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
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
        self.conv1_1_D = nn.Conv2d(64, 6, 3, padding=1)
        
        
        
        
# 4 module for subnetwork
        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(
            n_blocks=block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1)
        self.feature_res_2 = ResidualBlockPSP(
            n_blocks=block_config[1], in_channels=256, mid_channels=128, out_channels=512, stride=2, dilation=1)
        self.feature_dilated_res_1 = ResidualBlockPSP(
            n_blocks=block_config[2], in_channels=512, mid_channels=256, out_channels=1024, stride=1, dilation=2)
        self.feature_dilated_res_2 = ResidualBlockPSP(
            n_blocks=block_config[3], in_channels=1024, mid_channels=512, out_channels=2048, stride=1, dilation=4)
        

        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[
            6, 3, 2, 1], height=img_size_8, width=img_size_8)

        self.decode_feature = DecodePSPFeature(
            height=img_size, width=img_size, n_classes=n_classes)

        self.aux = AuxiliaryPSPlayers(
            in_channels=1024, height=img_size, width=img_size, n_classes=n_classes)
        
        self.apply(self.weight_init)
    def forward(self, rgb_input):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(rgb_input)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)
        
        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)
        
        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x_1 = F.log_softmax(self.conv1_1_D(x))
        
        y = self.feature_conv(rgb_input)
        y = self.feature_res_1(y)
        y = self.feature_res_2(y)
        y = self.feature_dilated_res_1(y)

        #output_aux = self.aux(x)  #  a part of Feature module into Aux module

        y = self.feature_dilated_res_2(y)

        y = self.pyramid_pooling(y)
        x_2 = self.decode_feature(y)
        output = torch.add(x_1, x_2)
        
        return output

class conv2DBatchNormRelu(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels,
							  kernel_size, stride, padding, dilation, bias=bias)
		self.batchnorm = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		# inplace=True, decrease memory consumption by not preserving input  
		
	def forward(self, x):
		x = self.conv(x)
		x = self.batchnorm(x)
		outputs = self.relu(x)

		return outputs

class FeatureMap_convolution(nn.Module):
	def __init__(self):
		super().__init__()

		in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 3, 64, 3, 2, 1, 1, False
		self.cbnr_1 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

		in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 64, 3, 1, 1, 1, False
		self.cbnr_2 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

		in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 128, 3, 1, 1, 1, False
		self.cbnr_3 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

	def forward(self, x):
		x = self.cbnr_1(x)
		x = self.cbnr_2(x)
		x = self.cbnr_3(x)
		outputs = self.maxpool(x)
		return outputs

class ResidualBlockPSP(nn.Sequential):
	def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
		super().__init__()

		# bottleNeckPSP
		self.add_module(
			"block1",
			bottleNeckPSP(in_channels, mid_channels,
						  out_channels, stride, dilation)
		)

		# bottleNeckIdentifyPSP
		for i in range(n_blocks - 1):
			self.add_module(
				"block" + str(i+2),
				bottleNeckIdentifyPSP(
					out_channels, mid_channels, stride, dilation)
			)

class conv2DBatchNorm(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels,
							  kernel_size, stride, padding, dilation, bias=bias)
		self.batchnorm = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x = self.conv(x)
		outputs = self.batchnorm(x)

		return outputs

class bottleNeckPSP(nn.Module):
	def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
		super().__init__()

		self.cbr_1 = conv2DBatchNormRelu(
			in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
		self.cbr_2 = conv2DBatchNormRelu(
			mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
		self.cb_3 = conv2DBatchNorm(
			mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		# skip concatenate
		self.cb_residual = conv2DBatchNorm(
			in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
		residual = self.cb_residual(x)
		return self.relu(conv + residual)

class bottleNeckIdentifyPSP(nn.Module):
	def __init__(self, in_channels, mid_channels, stride, dilation):
		super().__init__()

		self.cbr_1 = conv2DBatchNormRelu(
			in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
		self.cbr_2 = conv2DBatchNormRelu(
			mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
		self.cb_3 = conv2DBatchNorm(
			mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
		residual = x
		return self.relu(conv + residual)

class PyramidPooling(nn.Module):
	def __init__(self, in_channels, pool_sizes, height, width):
		super().__init__()

		self.height = height
		self.width = width

		out_channels = int(in_channels / len(pool_sizes))

		# pool_sizes: [6, 3, 2, 1]
		self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
		self.cbr_1 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
		self.cbr_2 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
		self.cbr_3 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
		self.cbr_4 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

	def forward(self, x):

		out1 = self.cbr_1(self.avpool_1(x))
		out1 = F.interpolate(out1, size=(
			self.height, self.width), mode="bilinear", align_corners=True)

		out2 = self.cbr_2(self.avpool_2(x))
		out2 = F.interpolate(out2, size=(
			self.height, self.width), mode="bilinear", align_corners=True)

		out3 = self.cbr_3(self.avpool_3(x))
		out3 = F.interpolate(out3, size=(
			self.height, self.width), mode="bilinear", align_corners=True)

		out4 = self.cbr_4(self.avpool_4(x))
		out4 = F.interpolate(out4, size=(
			self.height, self.width), mode="bilinear", align_corners=True)

		if (x.size(2) != out1.size(2) or out1.size(2) != out2.size(2)):
			print(x.size(), out1.size(), out2.size(), out3.size(), out4.size())
		 
		output = torch.cat([x, out1, out2, out3, out4], dim=1)
		
		return output

class DecodePSPFeature(nn.Module):
	def __init__(self, height, width, n_classes):
		super().__init__()

		self.height = height
		self.width = width

		self.cbr = conv2DBatchNormRelu(
			in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
		self.dropout = nn.Dropout2d(p=0.1)
		self.classification = nn.Conv2d(
			in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		x = self.cbr(x)
		x = self.dropout(x)
		x = self.classification(x)
		output = F.interpolate(
			x, size=(self.height, self.width), mode="bilinear", align_corners=True)
		return output

class AuxiliaryPSPlayers(nn.Module):
	def __init__(self, in_channels, height, width, n_classes):
		super().__init__()

		self.height = height
		self.width = width

		self.cbr = conv2DBatchNormRelu(
			in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
		self.dropout = nn.Dropout2d(p=0.1)
		self.classification = nn.Conv2d(
			in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		x = self.cbr(x)
		x = self.dropout(x)
		x = self.classification(x)
		output = F.interpolate(
			x, size=(self.height, self.width), mode="bilinear", align_corners=True)
		return output
