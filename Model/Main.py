from Model.Positional_Encoding import *
from Model.AddAndNorm import *
from Model.Encoder import *
from Model.Decoder import *

import random
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy

def generate_encoder_mask(input):
	output = (input != 0).cuda().byte().unsqueeze(-2)
	return output

def generate_subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	subsequent_mask = numpy.triu(numpy.ones(attn_shape), k=1).astype("uint8")
	return (torch.from_numpy(subsequent_mask) == 0).cuda()

def generate_decoder_mask(input):
	"Create a mask to hide padding and future words."
	output = (input != 0).unsqueeze(-2)
	output = output & Variable(generate_subsequent_mask(input.size(-1)).type_as(output.data))
	return output

class TransformerModel(nn.Module):
	"Base transformer model class"
	def __init__(self, d_model, vocab_size, max_len, n_layers, n_head, d_q, d_k, d_v, d_inner, P_drop):
		super(TransformerModel, self).__init__()

		# Encoder layers
		self.encoder_embedding = nn.Embedding(vocab_size, d_model)
		self.encoder_positional_encoding = PositionalEncoding(d_model, max_len)
		self.encoder_add_and_norm = AddAndNorm(d_model, P_drop)
		self.encoder = Encoder(n_layers, n_head, d_q, d_k, d_v, d_model, d_inner, max_len, P_drop)

		# Decoder layers
		self.decoder_embedding = nn.Embedding(vocab_size, d_model)
		self.decoder_positional_encoding = PositionalEncoding(d_model, max_len)
		self.decoder_add_and_norm = AddAndNorm(d_model, P_drop)
		self.decoder = Decoder(n_layers, n_head, d_q, d_k, d_v, d_model, d_inner, max_len, P_drop)

		# Output layers
		self.linear = nn.Linear(d_model, vocab_size)

	def forward(self, src, tgt):

		# Encoder Word Embedding
		encoder_embedding_output = self.encoder_embedding(src)

		# Encoder Positional Encoding
		encoder_positional_encoding_output = self.encoder_positional_encoding(encoder_embedding_output)

		# Initial Encoder Add & Norm Operation (Not shown on Figure 1 but exists)
		encoder_input = self.encoder_add_and_norm(encoder_embedding_output, encoder_positional_encoding_output)

		# Generate Encoder Mask
		encoder_mask = generate_encoder_mask(src)

		# Encoder
		encoder_output = self.encoder(encoder_input, encoder_mask)

		# Decoder Word Embedding
		decoder_embedding_output = self.decoder_embedding(tgt)

		# Decoder Positional Encoding
		decoder_positional_encoding_output = self.decoder_positional_encoding(decoder_embedding_output)

		# Initial Decoder Add & Norm Operation (Not shown on Figure 1 but exists)
		decoder_input = self.decoder_add_and_norm(decoder_embedding_output, decoder_positional_encoding_output)

		# Generate Decoder Mask
		decoder_mask = generate_decoder_mask(tgt)

		# Decoder
		decoder_output = self.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask)

		# Linear
		linear_output = self.linear(decoder_output)

		return linear_output
