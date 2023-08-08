import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class ScaledDotProductAttention(nn.Module):
	"Scaled Dot-Product Attention"
	def __init__(self, d_k):
		super(ScaledDotProductAttention, self).__init__()
		self.d_k = d_k

	def forward(self, q, k, v, mask):
		# MatMul
		matmul_out = torch.bmm(q, k.transpose(1,2))

		# Scale
		scale_out = matmul_out / math.sqrt(self.d_k)

		# Mask
		masked_out = scale_out * mask

		# SoftMax
		softmax_out = F.softmax(masked_out, -1)

		# Final MatMul
		matmul_out = torch.bmm(softmax_out, v)

		return matmul_out
