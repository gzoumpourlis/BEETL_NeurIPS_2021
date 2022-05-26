import os
import sys
sys.dont_write_bytecode = True
import random
import datetime

import train
from args_utils import create_exp_args

random.seed(2)

# Create the experiment's name string, containing date and time
now = datetime.datetime.now()
exp_name = now.strftime("%Y_%m_%d-%H%M%S")

exp_args_dict = {
		'finetune':				True,

		# Training across datasets or across sessions
		'mode':					'cross_session',

		# Aligning data using covariances, before ("pre") or during ("batch") training
		'pre_align':				False,
		'batch_align':				True,
		'coef_cov_mix':				0.5,

		# Train set
		'dataset_name':				'Cybathlon',
		'batch_size':				64,

		# Test set
		'dataset_test_name':		'Cybathlon',
		'batch_size_val':			64,

		# Domain adaptation hyperparameters
		'use_da':				True,
		'da_type':				'dann',
		'coef_domain':				1.0,
		'da_program':				'fixed',
		'batch_size_da':			64,
		###################################################################
		# Network architecture
		'net_cls':				'eegnetv4',
		'dropout':				0.3,

		# Load pretrained checkpoint
		'load_ckpt':				True,
		'ckpt_file':				os.path.join('checkpoints', 'net_best_pretrained.pth'),
		'checkpoint':				'best',

		# Use warmup period or scheduler for learning rate
		'use_scheduler':			False,
		'warmup':				False,
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
		'max_epochs':				130,
		'warmup_epochs':			20,
		'optim':				'adamw',
		'lr':					0.0001,
		'momentum':				0.9,
		'weight_decay':				0.0005,
		'coef_cls':				1.0,
		#############################################
		# Log experiment using "Weights and Biases" (wandb) package
		'wandb_log':				False,
		#############################################
		'comment':				''
		}

def main(exp_args_dict):

	args = create_exp_args(exp_args_dict=exp_args_dict)
	acc = train.main(args)
	print('\n\nVal accuracy: {:.2f}\n'.format(acc))

if __name__=='__main__':
	main(exp_args_dict)
