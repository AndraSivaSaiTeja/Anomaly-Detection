
#import segmentation_models_pytorch as smp
import torch
from torch import nn
from pathlib import Path
import sys
	

class Dn0(nn.Module):
	def __init__(self):
		super(Dn0, self).__init__()
		self.batchnorm = nn.Sequential(
			torch.nn.BatchNorm2d(3)
		)
		self.encoder1 = nn.Sequential(
			nn.Conv2d(3, 8, 3, padding=1),
			#nn.LeakyReLU(0.2),
			nn.Conv2d(8, 8, 3, padding=1),
			#torch.nn.BatchNorm2d(8)
			nn.LeakyReLU(0.2)
			)

		self.encoder2 = nn.Sequential(
			nn.Conv2d(8, 16, 3, padding=1),
			#nn.LeakyReLU(0.2),
			nn.Conv2d(16, 16, 3, padding=1),
			#torch.nn.BatchNorm2d(16)
			nn.LeakyReLU(0.2),
			)

		self.encoder3 = nn.Sequential(
			nn.Conv2d(16, 8, 3, padding=1),
			#nn.LeakyReLU(0.2),
			nn.Conv2d(8, 8, 3, padding=1),
			#torch.nn.BatchNorm2d(8)
			nn.LeakyReLU(0.2)
			)
		self.encoder4 = nn.Sequential(
			nn.Conv2d(8, 8, 3, padding=1),
			nn.Conv2d(8, 3, 3, padding=1),
			nn.LeakyReLU(0.2)
			#torch.nn.BatchNorm2d(3)
			)
		self.pool = nn.MaxPool2d(2,stride=2)

	def forward(self, x):
		#x = self.batchnorm(x)
		out1 = self.encoder1(x)
		#pool1 = self.pool(out1) # 128 * 128
		out2 = self.encoder2(out1)
		#pool2 = self.pool(out2) # 64 * 64
		out3 = self.encoder3(out2)
		#pool3 = self.pool(out3) # 32 * 32
		out4 = self.encoder4(out3)
		#pool4 = self.pool(out4)
		return out4

class Dn1(nn.Module):
	def __init__(self):
		super(Dn1, self).__init__()
		self.batchnorm = nn.Sequential(
			torch.nn.BatchNorm2d(64)
		)
		self.encoder1 = nn.Sequential(
			nn.Conv2d(64, 96, 3, padding=1),
			#nn.LeakyReLU(0.2),
			nn.Conv2d(96, 96, 3, padding=1),
			#torch.nn.BatchNorm2d(96)
			nn.LeakyReLU(0.2)
			)

		self.encoder2 = nn.Sequential(
			nn.Conv2d(96, 128, 3, padding=1),
			#nn.LeakyReLU(0.2),
			nn.Conv2d(128, 128, 3, padding=1),
			#torch.nn.BatchNorm2d(128)
			nn.LeakyReLU(0.2)
			)

		self.encoder3 = nn.Sequential(
			nn.Conv2d(128, 96, 3, padding=1),
			#nn.LeakyReLU(0.2),
			nn.Conv2d(96, 96, 3, padding=1),
			#torch.nn.BatchNorm2d(96)
			nn.LeakyReLU(0.2)
			)
		self.encoder4 = nn.Sequential(
			nn.Conv2d(96, 64, 3, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(64, 64, 3, padding=1),
			torch.nn.BatchNorm2d(64))
		self.pool = nn.MaxPool2d(2,stride=2)

	def forward(self, x):
		#x = self.batchnorm(x)
		out1 = self.encoder1(x)
		#pool1 = self.pool(out1) # 128 * 128
		out2 = self.encoder2(out1)
		#pool2 = self.pool(out2) # 64 * 64
		out3 = self.encoder3(out2)
		#pool3 = self.pool(out3) # 32 * 32
		out4 = self.encoder4(out3)
		#pool4 = self.pool(out4)
		return out4
		
		
		
class Dn2(nn.Module):
	def __init__(self):
		super(Dn2, self).__init__()
		self.batchnorm = nn.Sequential(
			torch.nn.BatchNorm2d(128)
		)
		self.encoder1 = nn.Sequential(
			nn.Conv2d(128, 160, 3, padding=1),
			#nn.LeakyReLU(0.2),
			nn.Conv2d(160, 160, 3, padding=1),
			#torch.nn.BatchNorm2d(160)
			nn.LeakyReLU(0.2)
			)

		self.encoder2 = nn.Sequential(
			nn.Conv2d(160, 192, 3, padding=1),
			#nn.LeakyReLU(0.2),
			nn.Conv2d(192, 192, 3, padding=1),
			#torch.nn.BatchNorm2d(192)
			nn.LeakyReLU(0.2)
			)

		self.encoder3 = nn.Sequential(
			nn.Conv2d(192, 160, 3, padding=1),
			#nn.LeakyReLU(0.2),
			nn.Conv2d(160, 160, 3, padding=1),
			#torch.nn.BatchNorm2d(160)
			nn.LeakyReLU(0.2)
			)
		self.encoder4 = nn.Sequential(
			nn.Conv2d(160, 128, 3, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(128, 128, 3, padding=1),
			torch.nn.BatchNorm2d(128))
		self.pool = nn.MaxPool2d(2,stride=2)

	def forward(self, x):
		#x = self.batchnorm(x)
		out1 = self.encoder1(x)
		#pool1 = self.pool(out1) # 128 * 128
		out2 = self.encoder2(out1)
		#pool2 = self.pool(out2) # 64 * 64
		out3 = self.encoder3(out2)
		#pool3 = self.pool(out3) # 32 * 32
		out4 = self.encoder4(out3)
		#pool4 = self.pool(out4)
		return out4
		
		
class Dn3(nn.Module):
	def __init__(self):
		super(Dn3, self).__init__()
		self.batchnorm = nn.Sequential(
			torch.nn.BatchNorm2d(256)
		)
		self.encoder1 = nn.Sequential(
			nn.Conv2d(256, 288, 3, padding=1),
			#nn.LeakyReLU(0.2),
			nn.Conv2d(288, 288, 3, padding=1),
			#torch.nn.BatchNorm2d(288)
			nn.LeakyReLU(0.2)
			)

		self.encoder2 = nn.Sequential(
			nn.Conv2d(288, 310, 3, padding=1),
			#nn.LeakyReLU(0.2),
			nn.Conv2d(310, 310, 3, padding=1),
			#torch.nn.BatchNorm2d(310)
			nn.LeakyReLU(0.2)
			)

		self.encoder3 = nn.Sequential(
			nn.Conv2d(310, 288, 3, padding=1),
			#nn.LeakyReLU(0.2),
			nn.Conv2d(288, 288, 3, padding=1),
			#torch.nn.BatchNorm2d(288)
			nn.LeakyReLU(0.2)
			)
		self.encoder4 = nn.Sequential(
			nn.Conv2d(288, 256, 3, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(256, 256, 3, padding=1),
			torch.nn.BatchNorm2d(256))
		self.pool = nn.MaxPool2d(2,stride=2)

	def forward(self, x):
		#x = self.batchnorm(x)
		out1 = self.encoder1(x)
		#pool1 = self.pool(out1) # 128 * 128
		out2 = self.encoder2(out1)
		#pool2 = self.pool(out2) # 64 * 64
		out3 = self.encoder3(out2)
		#pool3 = self.pool(out3) # 32 * 32
		out4 = self.encoder4(out3)
		#pool4 = self.pool(out4)
		return out4
