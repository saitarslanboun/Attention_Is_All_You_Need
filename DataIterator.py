from torch.utils.data.dataset import Dataset

import codecs
import torch

def collate_fn(batch):
	# Reducing the data length by removing unnecessary pads in order to reduce computational complexity
	src_token_ids, src_token_length, tgt_token_ids, tgt_token_length = zip(*batch)

	# Identify the length of the sample with maximum number of non-PAD tokens
	max_src_token_length = max(src_token_length)
	max_tgt_token_length = max(tgt_token_length)

	# Convert token_ids from list to torch tensor
	src_tokens = torch.LongTensor(src_token_ids)
	tgt_tokens = torch.LongTensor(tgt_token_ids)

	# Chunk the tokens tensor into the length of max_token_length
	src_tokens = src_tokens[:, :max_src_token_length]
	tgt_tokens = tgt_tokens[:, :max_tgt_token_length]

	return src_tokens, tgt_tokens

class LoadData(Dataset):
	def __init__(self, src, tgt, vocab, max_len=512):
		self.src = codecs.open(src, encoding='utf-8').readlines()
		self.tgt = codecs.open(tgt, encoding='utf-8').readlines()
		self.vocab = vocab
		self.max_len = max_len

	def __getitem__(self, index):

		# Loading source and target sentences from index
		src = self.src[index]
		tgt = self.tgt[index]

		src_tokens = src.strip().split()
		tgt_tokens = tgt.strip().split()

		# Preparing token ids
		src_token_ids = []
		for token in src_tokens:
			try:
				src_token_ids.append(self.vocab[token])
			except:
				src_token_ids.append(self.vocab['<unk>'])

		tgt_token_ids = []
		for token in tgt_tokens:
			try:
				tgt_token_ids.append(self.vocab[token])
			except:
				tgt_token_ids.append(self.vocab['<unk>'])

		# Add start and end tokens
		src_token_ids = [self.vocab['<s>']] + src_token_ids + [self.vocab['</s>']]
		tgt_token_ids = [self.vocab['<s>']] + tgt_token_ids + [self.vocab['</s>']]

		# If the length of token_ids is greater than maximum length, chunk it to maximum size
		src_token_ids = src_token_ids[:self.max_len]
		tgt_token_ids = tgt_token_ids[:self.max_len]

		# Keep length of the token lists
		src_token_length = len(src_token_ids)
		tgt_token_length = len(tgt_token_ids)

		# Pad token ids
		src_token_ids += [self.vocab['<unk>']] * (self.max_len - src_token_length)
		tgt_token_ids += [self.vocab['<unk>']] * (self.max_len - tgt_token_length)

		return src_token_ids, src_token_length, tgt_token_ids, tgt_token_length

	def __len__(self):
		return len(self.src)
