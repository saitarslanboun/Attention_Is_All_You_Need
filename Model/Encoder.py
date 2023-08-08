from Model.EncoderLayer import *

import torch.nn as nn

class Encoder(nn.Module):
	"Encoder Layer"
	def __init__(self, n_layers, n_head, d_q, d_k, d_v, d_model, d_inner, max_len, P_drop):
		super(Encoder, self).__init__()
		self.encoder_layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_q, d_k, d_v, P_drop) for _ in range(n_layers)])

	def forward(self, input, mask):
		encoder_output = input

		for encoder_layer in self.encoder_layer_stack:
			encoder_output = encoder_layer(encoder_output, mask)

		return encoder_output
