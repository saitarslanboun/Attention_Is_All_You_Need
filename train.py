from Model.Main import *
from DataIterator import *
from torch.utils.data import DataLoader
from Optimizer import *
from tqdm import tqdm

import argparse
import json
import os
import torch.nn.functional as F
import torch.optim as optim

def calc_loss(pred, truth):
	# Reshape the tensors before calculating the loss
	pred = pred.contiguous().view(-1, pred.shape[-1])
	truth = truth.contiguous().view(-1)

	# Cross entropy loss with label smoothing
	loss = F.cross_entropy(pred, truth, label_smoothing=opt.eps_ls)
	return loss

def calc_acc(pred, truth):
	# Get the argmax of prediction logits
	prediction = pred.argmax(-1)

	# Reshape the tensors before calculating the accuracy
	prediction = prediction.contiguous().view(-1)
	truth = truth.contiguous().view(-1)

	# Get the number of labels
	num_of_labels = prediction.shape[0]

	# Match the predicted labels with the ground truth
	prediction[prediction==0] = -1
	truth[truth==0] = -2
	acc = ((prediction == truth).sum().item() / num_of_labels) * 100

	return acc

def valid_iter(valid_data_iter):
	# Move model gradients to the validation state
	model.eval()

	# Validation progress bar
	progress_bar = tqdm(valid_data_iter, mininterval=1, desc="Validating", leave=False)

	# Keep Total Loss
	total_loss = 0

	# Keep Total Accuracy
	total_acc = 0

	# Start validation
	for num_of_steps, batch in enumerate(progress_bar):

		# prepare_data
		src, tgt = batch
		src = src.cuda()
		tgt = tgt.cuda()

		# Forward propagation
		pred = model(src, tgt[:, :-1])

		# Calculate Loss
		loss = calc_loss(pred, tgt[:, 1:])
		total_loss += loss.item()
		avg_loss = total_loss / (num_of_steps+1)

		# Calculate Accuracy
		acc = calc_acc(pred, tgt[:, 1:])
		total_acc += acc
		avg_acc = total_acc / (num_of_steps+1)

	return avg_loss, avg_acc

def train_iter(train_data_iter, valid_data_iter, log_train_file, log_valid_file):
	model.train()

	num_of_steps = 0

	loader_iter = iter(train_data_iter)

	# Keep Total Loss
	total_loss = 0

	# Keep Total Accuracy
	total_acc = 0

	# Stop training when the number of steps reaches to predefined training steps
	while num_of_steps < opt.train_steps:
		num_of_steps += 1

		# Get next batch
		try:
			current_batch = next(loader_iter)
		except:
			loader_iter = iter(train_data_iter)
			current_batch = next(loader_iter)

		# Prepare data
		src, tgt = current_batch
		src = src.cuda()
		tgt = tgt.cuda()

		# Forward propagation
		pred = model(src, tgt[:, :-1])

		# Calculate Loss
		loss = calc_loss(pred, tgt[:, 1:])
		total_loss += loss.item()
		avg_loss = total_loss / num_of_steps

		# Calculate Accuracy
		acc = calc_acc(pred, tgt[:, 1:])
		total_acc += acc
		avg_acc = total_acc / num_of_steps

		# Backward propagation
		loss.backward()

		# Update parameters and learning rate
		optimizer.step()

		# Set residual gradients to zero
		optimizer.optimizer.zero_grad()

		# Note keeping
		if num_of_steps % 10 == 0:
			description = "[Info] Training Steps: [" + str(num_of_steps) + "/" + str(opt.train_steps) + "] Avg Loss: " + str(avg_loss) + " Avg Acc: " + str(avg_acc)
			print (description)

		# Validate the model, and write updates to the log files
		if num_of_steps % 1000 == 0:
			description = "[Info] Validation after " + str(num_of_steps) + " steps."
			print (description)

			# Run validation script
			with torch.no_grad():
				valid_loss, valid_acc = valid_iter(valid_data_iter)

			# Write updates to the log files
			with open (log_train_file, "a") as log_tf, open(log_valid_file, "a") as log_vf:
				log_tf.write('{iteration}\t{loss: 8.5f}\n'.format(iteration=num_of_steps, loss=avg_loss))
				log_vf.write('{iteration}\t{loss: 8.5f}\t{acc: 3.3f}\n'.format(iteration=num_of_steps, loss=valid_loss, acc=valid_acc))

			# Move model back to training state
			model.train()

def train():
	# If checkpoint folder does not exist, generate a new one
	if not os.path.isdir(opt.checkpoint_dir):
		os.mkdir(opt.checkpoint_dir)

	# Initiate logging files
	log_train_file = os.path.join(opt.checkpoint_dir, "log_train.txt")
	log_valid_file = os.path.join(opt.checkpoint_dir, "log_valid.txt")

	# Log info
	print ('[Info] Training performance will be written to file: {}'.format(log_train_file))
	print ('[Info] Validation performance will be written to file: {}'.format(log_valid_file))

	# Write headers to train and validation logging files
	with open(log_train_file, 'w') as log_tf:
		log_tf.write('iteration\tloss\n')
	with open(log_valid_file, 'w') as log_vf:
		log_vf.write('iteration\tloss\taccuracy\n')

	# Prepare training data iterator
	train_data = LoadData(opt.train_src, opt.train_tgt, vocab, opt.max_len)
	train_data_iter = DataLoader(train_data, batch_size=opt.batch_size, collate_fn=collate_fn)

	# Prepare validation data iterator
	valid_data = LoadData(opt.valid_src, opt.valid_tgt, vocab, opt.max_len)
	valid_data_iter = DataLoader(valid_data, batch_size=opt.batch_size, collate_fn=collate_fn)

	# Start training
	train_iter(train_data_iter, valid_data_iter, log_train_file, log_valid_file)

if __name__ == '__main__':
	# Argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_src', help="Training source data", required=True)
	parser.add_argument('--train_tgt', help="Training target data", required=True)
	parser.add_argument('--valid_src', help="Validation source data", required=True)
	parser.add_argument('--valid_tgt', help="Validation target data", required=True)
	parser.add_argument('--vocab', help="Text vocabulary", required=True)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--checkpoint_dir', default='checkpoints')
	parser.add_argument('--max_len', type=int, default=512)
	parser.add_argument('--n_warmup_steps', type=int, default=4000)
	parser.add_argument('--N', type=int, default=6)
	parser.add_argument('--d_model', type=int, default=512)
	parser.add_argument('--d_ff', type=int, default=2048)
	parser.add_argument('--h', type=int, default=8)
	parser.add_argument('--d_k', type=int, default=64)
	parser.add_argument('--d_v', type=int, default=64)
	parser.add_argument('--P_drop', type=float, default=0.1)
	parser.add_argument('--eps_ls', type=float, default=0.1)
	parser.add_argument('--train_steps', type=int, default=279304)
	opt = parser.parse_args()

	# Prepare Vocabulary
	vocab_list = codecs.open(opt.vocab, encoding="utf-8").readlines()
	vocab_list = [val.strip().split("\t")[0] for val in vocab_list]
	vocab = dict()
	for ind in range(len(vocab_list)):
		vocab[vocab_list[ind]] = ind
	vocab_size = len(vocab)

	# Control parameters before building the model
	assert opt.d_model % opt.h == 0, 'd_model and h must be dividible!'
	assert opt.d_model % opt.d_k == 0, 'd_model and d_k must be dividible!'
	assert opt.d_model % opt.d_v == 0, 'd_model and d_v must be dividible!'

	# Build model
	model = TransformerModel(opt.d_model, vocab_size, opt.max_len, opt.N, opt.h, opt.d_k, opt.d_k, opt.d_v, opt.d_ff, opt.P_drop)

	# Carry model to CUDA environment for GPU-based computing
	model.cuda()

	# Set Optimizer
	optimizer = NoamOpt(opt.d_model, 2, opt.n_warmup_steps, optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	# Train the model
	train()

	# Get model state dictionary
	model_state_dictionary = model.state_dict()

	# Preapre checkpoint dictionary
	opt.vocab_size = vocab_size
	checkpoint = {"model_state_dictionary": model_state_dictionary, "parameters": opt}

	# Save final model state dictionary
	description = "[Info] Saving model checkpoint..."
	print (description)

	output_file_dir = os.path.join(opt.checkpoint_dir, "model.npz")
	torch.save(checkpoint, output_file_dir)
