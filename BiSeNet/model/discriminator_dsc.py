import torch.nn as nn
import torch.nn.functional as F
from model.DSC import depthwise_separable_conv as DSC


class DiscriminatorDSC(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(DiscriminatorDSC, self).__init__()

		self.conv1 = DSC(num_classes, ndf, kernel_size=4, padding=1)
		self.conv2 = DSC(ndf, ndf*2, kernel_size=4, padding=1)
		self.conv3 = DSC(ndf*2, ndf*4, kernel_size=4, padding=1)
		self.conv4 = DSC(ndf*4, ndf*8, kernel_size=4, padding=1)
		self.classifier = DSC(ndf*8, 1, kernel_size=4, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x
