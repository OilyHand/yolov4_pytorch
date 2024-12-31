import torch
import torch.nn as nn
import torch.nn.functional as F

class Activation(nn.Module):
'''
	Activation Funcitons
	- mish
	- leaky_relu
'''
	def __init__(self):
		super(activation, self).__init__()
	
	def mish(self, x):
		return F.mish(x)
	
	def leaky_relu(self, x):
		return F.leaky_relu(x)


ACTIVATIONS = {
	'leaky_relu' = Activation.leaky_relu(),
	'mish' = Activation.mish()
}



class CBA(nn.Module):
'''
	Convolutional Layer
	- sequence: conv2d(C) -> batchnorm(B) -> activation(A)
	- activation: mish(default) or leaky_relu
'''
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='mish'):
		super(CBA, self).__init__()

		self.__conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
			nn.BatchNorm2d(out_channels),
			ACTIVATIONS[activation]
		)

	def forward(self, x):
		return self.__conv(x)



class Residual(nn.Module):
'''
	Residual Block
'''
	def __init__(self, in_channels, out_channels, hidden_channels=None):
		super(Residual, self).__init__()

	if hidden_channels is None:
		hidden_channels = out_channels
	
	self.__block = nn.Sequential(
		CBA(in_channels, hidden_channels, 1)
		CBA(hidden_channels, out_channels, 3)
	)

	def forward(self, x):
		return x + self.__block(x)


class CSP(nn.Module):
'''
	Cross Stage Partial block
'''
	def __init__(self):
		pass
	
	def forward(x):
		pass


class CSPDarkNet53(nn.Module):
	def __init__(self):
		pass
	
	def forward(x):
		pass

if __name__ == "__main__":
	model = CSPDarkNet53()
