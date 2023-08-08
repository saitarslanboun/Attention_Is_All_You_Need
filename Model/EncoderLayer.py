from Model.MultiHead_Attention import *
from Model.AddAndNorm import *
from Model.Feed_Forward import *

import torch.nn as nn

class EncoderLayer(nn.Module):
	"Encoder Sub Layer"
	def __init__(self, d_model, d_inner, n_head, d_q, d_k, d_v, P_drop):
		super(EncoderLayer, self).__init__()
		self.multi_head_attention = MultiHeadAttention(n_head, d_model, d_q, d_k, d_v)
		self.add_and_norm_1 = AddAndNorm(d_model, P_drop)
		self.feed_forward = FeedForward(d_model, d_inner)
		self.add_and_norm_2 = AddAndNorm(d_model, P_drop)

	def forward(self, input, mask):
		# Multi-Head Attention
		multi_head_attention_output = self.multi_head_attention(input, input, input, mask=mask)

		# Add & Norm
		add_and_norm_output = self.add_and_norm_1(input, multi_head_attention_output)

		# Feed Forward
		feed_forward_output = self.feed_forward(add_and_norm_output)

		# Final Add & Norm
		add_and_norm_output = self.add_and_norm_2(add_and_norm_output, feed_forward_output)

		return add_and_norm_output
