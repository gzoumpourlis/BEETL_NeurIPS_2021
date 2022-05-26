import os
import sys
sys.dont_write_bytecode = True
import random
import datetime

import eval
from args_utils import create_exp_args

random.seed(1312)

# Create the experiment's name string, containing date and time
now = datetime.datetime.now()
exp_name = now.strftime("%Y_%m_%d-%H%M%S")

exp_args_dict = {
		'finetune':				False,

		# Training across datasets or across sessions
		'mode':					'cross_session',

		# Aligning data using covariances, before ("pre") or during ("batch") training
		'pre_align':				True,
		'batch_align':				False,
		'coef_cov_mix':				0.0,

		# Test set
		'dataset_test_name':			'Cybathlon',
		'dataset_test_phase':			'train',
		'batch_size_val':			64,

		###################################################################
		# Network architecture
		'net_cls':				'eegnetv4',
		'dropout':				0.1,

		# Load pretrained checkpoint
		'load_ckpt':				True,
		'ckpt_file':				os.path.join('checkpoints', 'net_best_pretrained.pth'),

		#############################################
		'cuda':					True,
		'exp_group':				exp_name,
		#############################################
		# Dataset preprocessing
		'fmin':					4.,
		'fmax':					38.,
		'sfreq':				100,
		'tmin':					0.0,
		'tmax':					4.0,

		# Apply Common Average Reference (CAR) on EEG signals
		'use_car':				True,
		# Add bipolar channels on EEG signals
		'use_bipolar':				True,
		#############################################
		# Training hyperparams
		'coef_cls':				1.0,
		#############################################
		# Log experiment using "Weights and Biases" (wandb) package
		'wandb_log':				False,
		#############################################
		'comment':				''
		}

def main(exp_args_dict):

	args = create_exp_args(exp_args_dict=exp_args_dict)
	acc = eval.main(args)
	print('\n\nVal accuracy: {:.2f}\n'.format(acc))

if __name__=='__main__':
	main(exp_args_dict)
