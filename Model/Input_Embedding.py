import torch.nn as nn

class InputEmbedding(nn.Module):
	def __init__(self, d_model, vocab_size):
		super(InputEmbedding, self).__init__()
		self.embedding = nn.Embedding(vocab_size, d_model)

	def forward(self, input):
		output = self.embedding(input)
		return output
