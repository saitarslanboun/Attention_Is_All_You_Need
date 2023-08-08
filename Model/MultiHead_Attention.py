from Model.ScaledDotProductAttention import *

import torch.nn as nn

class MultiHeadAttention(nn.Module):
	"Multi-Head Attention module"
	def __init__(self, n_head, d_model, d_q, d_k, d_v):
		super(MultiHeadAttention, self).__init__()

		# d_q and d_k must be equal
		assert d_q == d_k

		self.n_head = n_head
		self.d_q = d_q
		self.d_k = d_k
		self.d_v = d_v

		self.linear_q = nn.ModuleList([nn.Linear(int(d_model/n_head), d_q) for _ in range(n_head)])
		self.linear_k = nn.ModuleList([nn.Linear(int(d_model/n_head), d_k) for _ in range(n_head)])
		self.linear_v = nn.ModuleList([nn.Linear(int(d_model/n_head), d_v) for _ in range(n_head)])
		self.scaled_dot_product_attention = ScaledDotProductAttention(d_k)
		self.concat_linear = nn.Linear((d_v*n_head), d_model)

	def forward(self, q, k, v, mask):
		q = q.view(q.shape[0], q.shape[1], self.n_head, -1)
		k = k.view(k.shape[0], k.shape[1], self.n_head, -1)
		v = v.view(v.shape[0], v.shape[1], self.n_head, -1)

		# Iterate over number of heads
		sub_attns = []
		for ind in range(self.n_head):

			# Get sub queries, keys and values
			sub_q = q[:, :, ind, :]
			sub_k = k[:, :, ind, :]
			sub_v = v[:, :, ind, :]

			# Linear
			linear_q = self.linear_q[ind](sub_q)
			linear_k = self.linear_k[ind](sub_k)
			linear_v = self.linear_v[ind](sub_v)

			# Scaled Dot-Product Attention
			sub_attn_out = self.scaled_dot_product_attention(linear_q, linear_k, linear_v, mask)

			# append to sub_attns for the ConCat
			sub_attns.append(sub_attn_out)

		# ConCat
		concat_out = torch.cat(sub_attns, dim=-1)

		# Final Linear
		linear_out = self.concat_linear(concat_out)

		return linear_out
