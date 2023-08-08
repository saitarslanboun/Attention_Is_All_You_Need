from torch.utils.data.dataset import Dataset

import codecs
import torch
import json

def collate_fn(batch):
	# Reducing the data length by removing unnecessary pads in order to reduce computational complexity
	token_ids, token_length = zip(*batch)

	# Identify the length of the sample with maximum number of non-PAD tokens
	max_token_length = max(token_length)

	# Convert token_ids from list to torch tensor
	tokens = torch.LongTensor(token_ids)

	# Chunk the tokens tensor into the length of max_token_length
	tokens = tokens[:, :max_token_length]

	return tokens

class LoadData(Dataset):
	def __init__(self, txt, vocab, max_len=512):
		self.txt = codecs.open(txt, encoding='utf-8').readlines()
		self.vocab = json.load(open(vocab))
		self.max_len = max_len

	def __getitem__(self, index):

		# Loading txt from index
		txt = self.txt[index]

		# Preparing tokens
		tokens = txt.strip().split()
	#	token_ids = [self.vocab[val] for val in tokens]
		token_ids = []
		for val in tokens:
			try:
				token_ids.append(self.vocab[val])
			except:
				token_ids.append(self.vocab["<UNK>"])

		# Add [CLS] token at the beginning
		token_ids = [self.vocab['<CLS>']] + token_ids

		# If the length of token_ids is greater than maximum length, chunk it to maximum size
		token_ids = token_ids[:self.max_len]

		# Keep length of the token lists
		token_length = len(token_ids)

		# Pad token ids
		token_ids += [self.vocab['<PAD>']] * (self.max_len - token_length)

		return token_ids, token_length

	def __len__(self):
		return len(self.txt)
