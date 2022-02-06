import torch.nn as nn
import torch.nn.functional as F
from model.DSC import depthwise_separable_conv as DSC


class DiscriminatorDSC(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(DiscriminatorDSC, self).__init__()

		"""
		def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias, stride=2)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
		nin=ndf*4
		nout=ndf*8
		"""
		self.depthwise1 = nn.Conv2d(num_classes, num_classes, kernel_size=4, padding=1, groups=num_classes, bias=False, stride=2)
		self.pointwise1 = nn.Conv2d(num_classes, ndf, kernel_size=1, bias=False)
		#self.conv1 = DSC(num_classes, ndf, kernel_size=4, padding=1)
		self.depthwise2 = nn.Conv2d(ndf, ndf, kernel_size=4, padding=1, groups=ndf, bias=False, stride=2)
		self.pointwise2 = nn.Conv2d(ndf, ndf*2, kernel_size=1, bias=False)
		#self.conv2 = DSC(ndf, ndf*2, kernel_size=4, padding=1)
		self.depthwise3 = nn.Conv2d(ndf*2, ndf*2, kernel_size=4, padding=1, groups=ndf, bias=False, stride=2)
		self.pointwise3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=1, bias=False)
		#self.conv3 = DSC(ndf*2, ndf*4, kernel_size=4, padding=1)
		self.depthwise4 = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, padding=1, groups=ndf*4, bias=False, stride=2)
		self.pointwise4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=1, bias=False)
		#self.conv4 = DSC(ndf*4, ndf*8, kernel_size=4, padding=1)
		self.classifier = DSC(ndf*8, 1, kernel_size=4, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.depthwise1(x)
		x = self.pointwise1(x)
		#x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.depthwise2(x)
		x = self.pointwise2(x)
		#x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.depthwise3(x)
		x = self.pointwise3(x)
		#x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.depthwise4(x)
		x = self.pointwise4(x)
		#x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x
