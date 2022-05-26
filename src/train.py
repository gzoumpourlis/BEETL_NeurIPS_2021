import os
import sys
import wandb
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from src.initializers import *
from src.train_scripts import *
from skorch.history import History
from skorch.msg_print import PrintLog
from braindecode.models import net_wrapper
from braindecode.util import set_random_seeds

def main(args):

	# Logging experimental settings, using wandb
	if args.wandb_log:
		config = dict(
			mode=args.mode,
			learning_rate=args.lr,
			momentum=args.momentum,
			pre_align=args.pre_align,
			batch_align=args.batch_align,
			use_da=args.use_da,
			use_scheduler=args.use_scheduler,
			warmup=args.warmup,
			optim=args.optim,
			architecture=args.net_cls,
			dataset_id=args.dataset_name,
			experiment_id=args.exp_group,
			infra="Personal_PC",
		)
		if args.mode == 'cross_dataset':
			config.update({'dataset_test_id': args.dataset_test_name})
		if args.use_da:
			config.update({'da_type': args.da_type})
			config.update({'coef_domain': args.coef_domain})
		wandb.init(
			project="Beetl",
			notes="",
			config=config,
		)

	# Checking CUDA availability
	if args.cuda and torch.cuda.is_available():
		# torch.set_default_tensor_type('torch.cuda.FloatTensor')
		torch.backends.cudnn.benchmark = True
		device = 'cuda'
	else:
		# torch.set_default_tensor_type('torch.FloatTensor')
		device = 'cpu'
		pass

	# random seed to make results reproducible
	seed = 20200220
	set_random_seeds(seed=seed, cuda=args.cuda)

	################################################################################
	if args.finetune==False:
		# Training on Physionet
		# Network is randomly initialized, no pretrained checkpoint is used
		dataloader_train, _, dataset_info = init_dataset(args=args, dataset_name=args.dataset_name)

		# Unsupervised domain adaptation on Cybathlon training (labelled) set (class labels are not used)
		if args.use_da:
			dataloader_da, _ = init_Cybathlon_dataset(args, phase='train', shuffle=True,
																batch_size=args.batch_size_da)
			dataloader_da_iter = iter(dataloader_da)

		# Validation on Cybathlon training (labelled) set
		dataloader_val, _ = init_Cybathlon_dataset(args, phase='train', shuffle=False)
	else:
		# Finetuning on Cybathlon training set, network is initialized from a checkpoint
		# Checkpoint is pretrained on Physionet with unsup. dom. adapt. on Cybathlon training (labelled) set
		dataloader_train, dataset_info = init_Cybathlon_dataset(args, phase='train', shuffle=True, cov_mix=True)

		# Validation on BeetlMI Leaderboard training (labelled) set (only to observe acc/loss, not for training)
		dataloader_val, _ = init_BeetlMI_dataset(args, phase='train', shuffle=False, cov_mix=False)

		# Testing on Cybathlon test (unlabelled) set, after training has finished
		dataloader_test, _ = init_Cybathlon_dataset(args, phase='val', shuffle=False, cov_mix=False)

		# Unsupervised domain adaptation on Cybathlon test (unlabelled) set (class labels are not used)
		if args.use_da:
			dataloader_da, _ = init_Cybathlon_dataset(args, phase='val', shuffle=True,
																batch_size=args.batch_size_da, cov_mix=False)
			dataloader_da_iter = iter(dataloader_da)

		# Align the EEG data of the training (labelled) set, considering the covariances of the unlabelled set
		if args.batch_align:
			unlabelled_session_covs = dataloader_test.dataset.dataset_covs.copy()
			dataloader_train.dataset.dataset_unlabelled_covs = unlabelled_session_covs.copy()
	################################################################################

	N_batches = len(dataloader_train)
	N_batches_total = args.max_epochs*N_batches

	################################################################################
	# Net initialization

	net_cls = init_net(args, dataset_info)
	net = net_wrapper(net_cls)

	# Loading checkpoint of a pretrained model
	if args.load_ckpt:
		ckpt_filename = os.path.join(os.getcwd(), args.ckpt_file)
		net.load_state_dict(torch.load(ckpt_filename), strict=False)
		print('\nLoaded pretrained checkpoint: {}'.format(ckpt_filename))

	# In case of domain adaptation using the DANN [1] method, a domain discriminator is used
	# [1] : Y. Ganin and V. Lempitsky, "Unsupervised Domain Adaptation by Backpropagation", ICML 2015
	# https://proceedings.mlr.press/v37/ganin15.html
	if args.use_da:
		discr = init_discr(args)

	################################################################################
	# Initialize criteria and optimizers/schedulers

	# Criterion for classification loss
	criterion = init_criterion(args)

	# Optimizer for classification network
	optimizer = init_optimizer(args, net)

	if args.use_da:
		# Criterion for domain classification loss
		criterion_domain = F.binary_cross_entropy_with_logits
		# Optimizer for domain discriminator network
		optimizer_discr = init_optimizer(args, discr)

	# Use scheduler to change learning rate
	# Warmup stage: linear increase
	# Afterwards: cosine annealing
	if args.use_scheduler:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
															   T_max=args.max_epochs-args.warmup_epochs,
															   eta_min=0,
															   last_epoch=-1
															   )
		if args.use_da:
			scheduler_discr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_discr,
																   T_max=args.max_epochs - args.warmup_epochs,
																   eta_min=0,
																   last_epoch=-1
																   )
	################################################################################

	# History object, used to keep/print epoch statistics
	print_log = PrintLog().initialize()
	history = History()

	# Initialize best validation accuracy
	best_val_acc = 0

	# Initialize the total number of training iterations that have been done
	batch_cnt_total = 0

	# Set network and domain discriminator to training mode
	net.train()
	if args.use_da:
		discr.train()

	# Watch parameters and gradients of networks, using wandb
	if args.wandb_log:
		wandb.watch(net, criterion, log="all", log_freq=100) # https://docs.wandb.ai/ref/python/watch
		if args.use_da:
			wandb.watch(discr, criterion_domain, log="all", log_freq=100)

	print('\nTraining...\n')
	################################################################################

	for epoch in range(args.max_epochs):

		history.new_epoch()
		history.record('epoch', len(history))

		####################################################
		# If using a warmup period for the learning rate, then increase LR linearly after each epoch
		if args.warmup and epoch<=args.warmup_epochs:
			initial_lr = 0.00001
			warm_lr = args.lr
			step_lr = (warm_lr - initial_lr)/args.warmup_epochs
			# The LR that will be used for this epoch
			current_lr = initial_lr + step_lr*epoch
			for param_group in optimizer.param_groups:
				param_group['lr'] = current_lr
			if args.use_da:
				for param_group in optimizer_discr.param_groups:
					param_group['lr'] = current_lr
		####################################################
		# Set networks to training mode
		net.train()
		if args.use_da:
			discr.train()

		# Initialize number of correctly classified training samples, total training samples, and batches
		correct = 0
		total = 0
		batch_cnt = 0

		epoch_start_t = time.time()

		# Iterate over the training set
		for batch_idx, batch in enumerate(dataloader_train):
			batch_cnt += 1
			batch_cnt_total += 1
			# Setting value of domain loss coefficient, according to the DANN method (Ganin & Lempitsky)
			if args.use_da:
				if args.da_program=='exp_up':
					# Progress of training (from 0 to 1)
					domain_p = batch_cnt_total / N_batches_total
					# Increasing the domain loss coefficient according to the DANN method
					domain_loss_coef = args.coef_domain * (2 / (1 + np.exp(-10 * domain_p)) - 1)
				elif args.da_program=='fixed':
					# Fixed domain loss coefficient
					domain_loss_coef = args.coef_domain

			# Input shape: Batch x 1 x Channels x Time_length
			inputs = batch['eeg'].float().to(device)
			# Targets: class labels
			targets = batch['target'].to(device)

			if args.use_da:
				# Load batch from domain adaptation dataloader
				try:
					batch_da = next(dataloader_da_iter)
				except StopIteration:
					dataloader_da_iter = iter(dataloader_da)
					batch_da = next(dataloader_da_iter)
				inputs_da = batch_da['eeg'].float().to(device)
				targets_domain = torch.cat([torch.ones(inputs.shape[0]), torch.zeros(inputs_da.shape[0])]).to(device)
			##############################################

			if args.use_da:
				# Perform Domain Adaptation
				inputs_cat = torch.cat([inputs, inputs_da])
				out_dict = net(inputs_cat)
				out = out_dict['scores'][:inputs.shape[0]]
				out_domain = discr(out_dict['feats']).squeeze()
			else:
				# Without Domain Adaptation
				out_dict = net(inputs)
				out = out_dict['scores']

			# Classification loss
			loss_cls = criterion(out, targets)
			loss = args.coef_cls * loss_cls

			# Domain loss
			if args.use_da:
				loss_domain = criterion_domain(out_domain, targets_domain)
				loss = loss + domain_loss_coef * loss_domain
			##############################################
			# Do backward pass & optimizer step
			optimizer.zero_grad()
			if args.use_da:
				optimizer_discr.zero_grad()
			loss.backward()
			optimizer.step()
			if args.use_da:
				optimizer_discr.step()
			##############################################
			# Keep batch statistics (loss, acc)

			batch_loss = loss.item()
			if args.use_da:
				batch_loss_domain = loss_domain.item()
			del loss

			# Predicted class labels for training samples
			_, predicted = out.max(1)

			batch_correct = predicted.eq(targets).sum().item()
			batch_acc = 100.0 * batch_correct / targets.size(0)
			correct += batch_correct
			total += targets.size(0)

			# Log statistics on wandb
			if args.wandb_log:
				train_batch_log_dict = {
					'epoch': epoch+1,
					'batch_train_loss': batch_loss,
					'batch_train_acc': batch_acc,
				}
				if args.use_da:
					train_batch_log_dict.update({
						'batch_train_loss_domain': batch_loss_domain,
						'p': domain_loss_coef,
					})
				wandb.log(train_batch_log_dict)

			# Logging train batch statistics on history
			history.new_batch()
			history.record_batch('train_loss', batch_loss)
			history.record_batch('train_acc', batch_acc)
			history.record_batch('train_batch_size', len(targets))

		# Logging train epoch statistics on history
		history = update_history_on_phase_end(history=history, phase='train')
		####################################################
		# Validation phase
		dataloader_val, net, criterion, history, args, epoch_val_acc, pred_dict = val(dataloader_val, net,
																						  criterion, history,
																						  record_history=True,
																						  log_preds=True,
																						  args=args)
		epoch_end_t = time.time()

		# Logging validation epoch statistics on history
		history = update_history_on_phase_end(history=history, phase='val', get_acc=True, acc=epoch_val_acc)
		history.record('dur', epoch_end_t - epoch_start_t)
		# Print accuracy/loss for train/val phase, on epoch end
		print_log.on_epoch_end(history=history, verbose=True)
		####################################################

		# Perform scheduler step, when using LR scheduler
		if args.use_scheduler:
			if (args.warmup==False) or (args.warmup==True and epoch>args.warmup_epochs):
				scheduler.step()
				if args.use_da:
					scheduler_discr.step()

		# Log statistics in wandb
		if args.wandb_log:
			train_epoch_log_dict = {
									'epoch': epoch + 1,
									'train_loss': history[:, 'train_loss'][-1],
									'train_acc': history[:, 'train_acc'][-1],
									'val_loss': history[:, 'val_loss'][-1],
									'val_acc': history[:, 'val_acc'][-1],
			}
			if args.use_scheduler:
				if (args.warmup == True and epoch > args.warmup_epochs):
					# Warmup period finished. Use the LR of the scheduler
					last_lrs = scheduler.get_last_lr()
					last_lr = np.float64(last_lrs[0])
				else:
					# Warmup period unfinished. Use the LR that was manually computed
					last_lr = np.float64(current_lr)
				train_epoch_log_dict.update({'scheduler_lr': last_lr})
			wandb.log(train_epoch_log_dict)

		# In case of new max val. accuracy, save model checkpoint
		if epoch_val_acc > best_val_acc:
			best_val_acc = epoch_val_acc
			root_path = os.path.join(os.getcwd(), 'checkpoints')
			if not os.path.exists(root_path):
				os.makedirs(root_path, exist_ok=True)
			model_best_filename = os.path.join(root_path, 'net_best_{}.pth'.format(args.exp_group))
			torch.save(net.state_dict(), model_best_filename)
	######################################################################

	results_columns = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
	df = pd.DataFrame(history[:, results_columns], columns=results_columns, index=history[:, 'epoch'])
	val_accs = df['val_acc'].values
	max_val_acc = np.max(val_accs)

	# Log summary in wandb
	if args.wandb_log:
		wandb.run.summary["best_val_acc"] = max_val_acc
		wandb.finish()

	# Save model checkpoint on the last iteration
	root_path = os.path.join(os.getcwd(), 'checkpoints')
	if not os.path.exists(root_path):
		os.makedirs(root_path, exist_ok=True)
	model_last_filename = os.path.join(root_path, 'net_last_{}.pth'.format(args.exp_group))
	torch.save(net.state_dict(), model_last_filename)

	######################################################################
	# Evaluate on a dataset (done only after finetuning)

	if args.finetune:

		if args.checkpoint == 'best':
			print('Loading best checkpoint')
			net.load_state_dict(torch.load(model_best_filename), strict=False)
		elif args.checkpoint == 'last':
			print('Loading last checkpoint')
			net.load_state_dict(torch.load(model_last_filename), strict=False)
		net.eval()

		# Get predictions in a dictionary
		_, _, _, _, _, _, pred_dict = val(
			dataloader_test,
			net,
			criterion,
			history,
			record_history=False,
			log_preds=True,
			args=args)

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

	return max_val_acc