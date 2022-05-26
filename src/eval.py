import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from src.initializers import *
from src.train_scripts import *
from braindecode.models import net_wrapper
from braindecode.util import set_random_seeds

def main(args):

	# Checking CUDA availability
	if args.cuda and torch.cuda.is_available():
		torch.backends.cudnn.benchmark = True

	# random seed to make results reproducible
	seed = 20200220
	set_random_seeds(seed=seed, cuda=args.cuda)
	################################################################################
	if args.dataset_test_name=='BeetlMI':
		# BeetlMI Leaderboard dataset
		dataloader, dataset_info = init_BeetlMI_dataset(args, phase=args.dataset_test_phase, shuffle=False, cov_mix=False)
	elif args.dataset_test_name=='Cybathlon':
		if args.batch_align:
			if args.dataset_test_phase=='train':
				cov_mix = True
			else:
				cov_mix = False
		else:
			cov_mix = False

		# Cybathlon dataset
		dataloader, dataset_info = init_Cybathlon_dataset(args, phase=args.dataset_test_phase, shuffle=False, cov_mix=cov_mix)

		if args.batch_align and cov_mix:
			# Evaluating on Cybathlon training (labelled) set. Passing test set's covariances as well.
			dataloader_unlabelled, _ = init_Cybathlon_dataset(args, phase='val', shuffle=False, cov_mix=False)
			unlabelled_session_covs = dataloader_unlabelled.dataset.dataset_covs.copy()
			dataloader.dataset.dataset_unlabelled_covs = unlabelled_session_covs.copy()
	################################################################################
	# Net initialization

	net_cls = init_net(args, dataset_info)
	net = net_wrapper(net_cls)

	# Loading checkpoint of a pretrained model
	if args.load_ckpt:
		ckpt_filename = os.path.join(os.getcwd(), args.ckpt_file)
		net.load_state_dict(torch.load(ckpt_filename), strict=False)
		print('Loaded pretrained checkpoint: {}'.format(ckpt_filename))

	# Set network to eval mode
	net.eval()

	# Initialize criterion for classification loss
	criterion = init_criterion(args)

	################################################################################
	# Evaluate

	print('\nEvaluating...\n')

	_, _, _, _, _, acc, pred_dict = val(dataloader,
		net,
		criterion,
		history=[],
		record_history=False,
		log_preds=True,
		args=args)

	print('Accuracy on validation set: {:.2f}'.format(acc))

	pred_test = pred_dict['pred']
	pred_test = pred_test.astype(int)
	# class IDs 2 ("feet") and 3 ("rest") are merged in a single class (2, "other")
	pred_test[pred_test == 3] = 2

	# Save class label predictions in .txt file
	preds_folder = os.path.join(os.getcwd(), 'preds', args.exp_group)
	if not os.path.exists(preds_folder):
		os.makedirs(preds_folder, exist_ok=True)
	txt_filepath = os.path.join(preds_folder, 'answer.txt')
	np.savetxt(txt_filepath, pred_test, delimiter=',', fmt="%d")

	# Save dictionary containing predictions and some more metadata in .npy file
	npy_filepath = os.path.join(preds_folder, 'pred_dict.npy')
	np.save(npy_filepath, pred_dict)

	return acc
