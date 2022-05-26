import os
import json
import copy
import argparse

def parse_default_args():
	##########################################################################
	parser = argparse.ArgumentParser(description="Beetl Competition")
	parser.add_argument("--exp_group", default='',
						help="name of experiment folder, when executing a group of experiments")
	parser.add_argument("--comment", default='', help="comment to append on folder name")
	# ---------------------------------------------------------------------------------------------------------------- #
	parser.add_argument('--dataset_name', type=str, choices=('BCIC_IV_2a', 'Cho2017', 'PhysionetMI', 'BeetlMI', 'Cybathlon'), default='PhysionetMI',
						help="set training set name")
	parser.add_argument('--mode', type=str, choices=('cross_session', 'cross_dataset'), default='cross_session',
						help="set training mode, across datasets or across sessions")

	parser.add_argument('--fmin', type=float, default=4., help="set min frequency for bandpass filtering")
	parser.add_argument('--fmax', type=float, default=38., help="set max frequency for bandpass filtering")
	parser.add_argument('--tmin', type=float, default=0.0, help="set tmin for event cropping")
	parser.add_argument('--tmax', type=float, default=4.0, help="set tmax for event cropping")

	parser.add_argument('--subject', type=int, default=1, help="in case of LOSO, set ID of the only test subject")
	# ---------------------------------------------------------------------------------------------------------------- #
	parser.add_argument("--net_cls", default='eegnetv4', choices=('eegnetv4',), help="name of model")
	parser.add_argument("--coef_cls", type=float, default=1.0, help="coefficient of classification loss")
	parser.add_argument('--checkpoint', type=str, choices=('best', 'last'), default='best', help="model ckpt")
	parser.add_argument('--load_ckpt', dest='load_ckpt', action='store_true', help="load ckpt of pretrained model")
	parser.set_defaults(load_ckpt=False)
	parser.add_argument('--ckpt_file', type=str, default='', help="model ckpt file")
	# ---------------------------------------------------------------------------------------------------------------- #
	parser.add_argument('--max_epochs', type=int, default=25, help="max number of epochs")
	parser.add_argument('--warmup_epochs', type=int, default=0, help="number of warmup epochs")
	parser.add_argument('--optim', type=str, default='sgd', choices=('adam', 'adamw', 'sgd'), help="optimizer")
	parser.add_argument('--use_scheduler', dest='use_scheduler', action='store_true', help="use optim scheduler")
	parser.set_defaults(use_scheduler=False)
	parser.add_argument('--warmup', dest='warmup', action='store_true', help="use lr warmup")
	parser.set_defaults(warmup=False)
	parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
	parser.add_argument('--momentum', type=float, default=0.9, help="momentum value for optimisation")
	parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight decay for SGD")
	parser.add_argument('--batch_size', type=int, default=32, help="batch size for training")
	parser.add_argument('--batch_size_val', type=int, default=1, help="batch size for validation/test")
	parser.add_argument('--batch_size_da', type=int, default=32, help="batch size for domain adaptation")
	parser.add_argument('--dropout', type=float, default=0.0, help="dropout")
	# ---------------------------------------------------------------------------------------------------------------- #
	parser.add_argument('--finetune', dest='finetune', action='store_true',
						help="finetune a pretrained model")
	parser.set_defaults(finetune=False)
	parser.add_argument('--use_car', dest='use_car', action='store_true', help="perform Common Average Reference on EEG signals")
	parser.set_defaults(use_car=False)
	parser.add_argument('--use_bipolar', dest='use_bipolar', action='store_true', help="add bipolar channels on EEG signals")
	parser.set_defaults(use_bipolar=False)
	parser.add_argument('--pre_align', dest='pre_align', action='store_true', help="perform alignment using covariances")
	parser.set_defaults(pre_align=False)
	parser.add_argument('--batch_align', dest='batch_align', action='store_true', help="perform alignment using covariances")
	parser.set_defaults(batch_align=False)
	parser.add_argument('--coef_cov_mix', type=float, default=0.0, help="weight to mix covariance matrices")
	parser.add_argument('--use_da', dest='use_da', action='store_true', help="perform domain adaptation")
	parser.set_defaults(use_da=False)
	parser.add_argument("--da_type", default='dann', choices=('dann',), help="type of domain adaptation method")
	parser.add_argument("--coef_domain", type=float, default=1.0, help="coefficient of domain loss")
	parser.add_argument("--da_program", default='exp_up', choices=('exp_up', 'fixed'), help="domain adapt. program")
	parser.add_argument('--wandb_log', dest='wandb_log', action='store_true', help="log experiment on WANDB")
	parser.set_defaults(wandb_log=False)
	parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
	parser.set_defaults(cuda=False)
	######################################################################
	default_args = parser.parse_args()

	return default_args

def overwrite_default_args(exp_args, default_args):

	for arg_key in vars(exp_args):
		arg_val = getattr(exp_args, arg_key)
		setattr(default_args, arg_key, arg_val)

	args = copy.deepcopy(default_args)

	return args

def dict_to_namespace(args):
	
	# Write arguments from dictionary to JSON
	json_path = write_args_json(args)

	# Read arguments from JSON to Namespace
	parser = argparse.ArgumentParser()
	with open(json_path, 'rt') as f:
		args_namespace = argparse.Namespace()
		args_namespace.__dict__.update(json.load(f))
		exp_args = parser.parse_args(namespace=args_namespace)

	return exp_args

def write_args_json(args):

	json_folder = os.path.join(os.getcwd(), 'json')
	if not os.path.exists(json_folder):
		os.makedirs(json_folder, exist_ok=True)

	filename = 'exp_args_{}.json'.format(args['exp_group'])
	json_path = os.path.join(json_folder, filename)

	with open(json_path, 'w') as out:
		json.dump(args, out)

	return json_path

def create_exp_args(exp_args_dict, subject_id=1):

	# default args
	default_args_namespace = parse_default_args()

	# exp args
	exp_args_dict.update({'subject': subject_id})
	exp_args_namespace = dict_to_namespace(exp_args_dict)

	# overwrite default with exp args
	args = overwrite_default_args(exp_args_namespace, default_args_namespace)

	return args