import torch

class Nested_UNet_3D(torch.nn.Module):
	def __init__(self, Starting_Channel, Deep_Supervision_Switch):
		super().__init__()
		self.Starting_Channel = Starting_Channel
		self.Deep_Supervision_Switch = Deep_Supervision_Switch

		n_channels = [1,self.Starting_Channel,self.Starting_Channel*2,self.Starting_Channel*4,self.Starting_Channel*8,self.Starting_Channel*16]

		# First Diagonal
		self.conv3d_layer_0_0 = conv_block(n_channels[0], n_channels[1])
		self.down_conv3d_layer_0_0 = down_conv_block(n_channels[1])
		self.conv3d_layer_1_0 = conv_block(n_channels[1], n_channels[2])
		self.down_conv3d_layer_1_0 = down_conv_block(n_channels[2])
		self.conv3d_layer_2_0 = conv_block(n_channels[2], n_channels[3])
		self.down_conv3d_layer_2_0 = down_conv_block(n_channels[3])
		self.conv3d_layer_3_0 = conv_block(n_channels[3], n_channels[4])
		self.down_conv3d_layer_3_0 = down_conv_block(n_channels[4])
		self.conv3d_layer_4_0 = conv_block(n_channels[4], n_channels[5])

		# Second Diagonal
		self.transposed_conv3d_0_1 = transposed_conv_block(n_channels[1])
		self.conv3d_layer_0_1 = conv_block(2 * n_channels[1], n_channels[1])
		self.transposed_conv3d_1_1 = transposed_conv_block(n_channels[2])
		self.conv3d_layer_1_1 = conv_block(2 * n_channels[2], n_channels[2])
		self.transposed_conv3d_2_1 = transposed_conv_block(n_channels[3])
		self.conv3d_layer_2_1 = conv_block(2 * n_channels[3], n_channels[3])
		self.transposed_conv3d_3_1 = transposed_conv_block(n_channels[4])
		self.conv3d_layer_3_1 = conv_block(2 * n_channels[4], n_channels[4])

		# Third Diagonal
		self.transposed_conv3d_0_2 = transposed_conv_block(n_channels[1])
		self.conv3d_layer_0_2 = conv_block(3 * n_channels[1], n_channels[1])
		self.transposed_conv3d_1_2 = transposed_conv_block(n_channels[2])
		self.conv3d_layer_1_2 = conv_block(3 * n_channels[2], n_channels[2])
		self.transposed_conv3d_2_2 = transposed_conv_block(n_channels[3])
		self.conv3d_layer_2_2 = conv_block(3 * n_channels[3], n_channels[3])

		# Fourth Diagonal
		self.transposed_conv3d_0_3 = transposed_conv_block(n_channels[1])
		self.conv3d_layer_0_3 = conv_block(4 * n_channels[1], n_channels[1])
		self.transposed_conv3d_1_3 = transposed_conv_block(n_channels[2])
		self.conv3d_layer_1_3 = conv_block(4 * n_channels[2], n_channels[2])

		# Fifth Diagonal
		self.transposed_conv3d_0_4 = transposed_conv_block(n_channels[1])
		self.conv3d_layer_0_4 = conv_block(5 * n_channels[1], n_channels[1])

		# Final Node
		self.conv_out = torch.nn.Conv3d(n_channels[1], n_channels[0], kernel_size=3, padding=1)

	def forward(self, x):
		x0_0 = self.conv3d_layer_0_0(x)
		x1_0 = self.conv3d_layer_1_0(self.down_conv3d_layer_0_0(x0_0))
		x0_1 = self.conv3d_layer_0_1(torch.cat((x0_0, self.transposed_conv3d_0_1(x1_0)), 1))

		x2_0 = self.conv3d_layer_2_0(self.down_conv3d_layer_1_0(x1_0))
		x1_1 = self.conv3d_layer_1_1(torch.cat((x1_0, self.transposed_conv3d_1_1(x2_0)), 1))
		x0_2 = self.conv3d_layer_0_2(torch.cat((x0_0, x0_1, self.transposed_conv3d_0_2(x1_1)), 1))

		x3_0 = self.conv3d_layer_3_0(self.down_conv3d_layer_2_0(x2_0))
		x2_1 = self.conv3d_layer_2_1(torch.cat((x2_0, self.transposed_conv3d_2_1(x3_0)), 1))
		x1_2 = self.conv3d_layer_1_2(torch.cat((x1_0, x1_1, self.transposed_conv3d_1_2(x2_1)), 1))
		x0_3 = self.conv3d_layer_0_3(torch.cat((x0_0, x0_1, x0_2, self.transposed_conv3d_0_3(x1_2)), 1))

		x4_0 = self.conv3d_layer_4_0(self.down_conv3d_layer_3_0(x3_0))
		x3_1 = self.conv3d_layer_3_1(torch.cat((x3_0, self.transposed_conv3d_3_1(x4_0)), 1))
		x2_2 = self.conv3d_layer_2_2(torch.cat((x2_0, x2_1, self.transposed_conv3d_2_2(x3_1)), 1))
		x1_3 = self.conv3d_layer_1_3(torch.cat((x1_0, x1_1, x1_2, self.transposed_conv3d_1_3(x2_2)), 1))
		x0_4 = self.conv3d_layer_0_4(torch.cat((x0_0, x0_1, x0_2, x0_3, self.transposed_conv3d_0_4(x1_3)), 1))

		if self.Deep_Supervision_Switch:
			xFinal_0_1 = self.conv_out(x0_1)
			xFinal_0_2 = self.conv_out(x0_2)
			xFinal_0_3 = self.conv_out(x0_3)
			xFinal_0_4 = self.conv_out(x0_4)
			return xFinal_0_1, xFinal_0_2, xFinal_0_3, xFinal_0_4
		else:
			xFinal = self.conv_out(x0_4)
			return xFinal

class conv_block(torch.nn.Module):
	def __init__(self, channels_in, channels_out):
		super().__init__()
		self.conv1 = torch.nn.Conv3d(channels_in, channels_out, kernel_size=3, padding=1)
		self.bn1 = torch.nn.BatchNorm3d(channels_out)
		self.relu1 = torch.nn.ReLU()
		self.conv2 = torch.nn.Conv3d(channels_out, channels_out, kernel_size=3, padding=1)
		self.bn2 = torch.nn.BatchNorm3d(channels_out)
		self.relu2 = torch.nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu2(x)
		return x

class down_conv_block(torch.nn.Module):
	def __init__(self, channels_out):
		super().__init__()
		self.conv_down = torch.nn.Conv3d(channels_out, channels_out, 2, stride=2)
		self.bn = torch.nn.BatchNorm3d(channels_out)
		self.relu = torch.nn.ReLU()

	def forward(self, x):
		x = self.conv_down(x)
		x = self.bn(x)
		x = self.relu(x)
		return x

class transposed_conv_block(torch.nn.Module):
	def __init__(self, channels_out):
		super().__init__()
		self.conv_transpose = torch.nn.ConvTranspose3d(2 * channels_out, channels_out, kernel_size=2, stride=2)
		self.bn = torch.nn.BatchNorm3d(channels_out)
		self.relu = torch.nn.ReLU()

	def forward(self, x):
		x = self.conv_transpose(x)
		x = self.bn(x)
		x = self.relu(x)
		return x
