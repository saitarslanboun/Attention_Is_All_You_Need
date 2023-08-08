import torch.nn as nn
import torch.nn.functional as F
import torch

import torch

class FeedForward(nn.Module):
	"Implements FFN equation."
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(FeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)
		self.act = nn.ReLU()

	def forward(self, input):
		output = self.w_1(input)
		output = self.act(output)
		output = self.dropout(output)
		output = self.w_2(output)
		return output
