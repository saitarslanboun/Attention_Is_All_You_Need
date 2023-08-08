from torch.autograd import Variable

import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
	"Implement the Positional Encoding function."
	def __init__(self, d_model, max_len=512):
		super(PositionalEncoding, self).__init__()
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.pe = pe

	def forward(self, input):
		#output = Variable(self.pe[:, :input.size(1)], requires_grad=False).cuda()
		output = self.pe[:, :input.size(1)].cuda()
		return output
