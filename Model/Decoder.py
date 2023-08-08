from Model.DecoderLayer import *

import torch.nn as nn

class Decoder(nn.Module):
	"Decoder Layer"
	def __init__(self, n_layers, n_head, d_q, d_k, d_v, d_model, d_inner, max_len, P_drop):
		super(Decoder, self).__init__()
		self.decoder_layer_stack = nn.ModuleList([DecoderLayer(d_model, d_inner, n_head, d_q, d_k, d_v, P_drop) for _ in range(n_layers)])

	def forward(self, input, encoder_output, encoder_mask, decoder_mask):
		decoder_output = input

		for decoder_layer in self.decoder_layer_stack:
			decoder_output = decoder_layer(decoder_output, encoder_output, encoder_mask, decoder_mask)

		return decoder_output
