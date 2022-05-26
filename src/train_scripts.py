import time
import torch
import numpy as np
import importlib.util

def val(dataloader_val, net, criterion, history, record_history, log_preds, args):

	if args.cuda:
		device = 'cuda'
	else:
		device = 'cpu'

	net.eval()
	val_loss = 0
	correct = 0
	total = 0

	if log_preds:
		target_list = list()
		pred_list = list()
		feat_list = list()

	for batch_idx, batch in enumerate(dataloader_val):
		inputs = batch['eeg'].float().to(device)
		targets = batch['target'].to(device)

		with torch.no_grad():
			out_dict = net(inputs)
			out = out_dict['scores']
			loss = criterion(out, targets)

		epoch_loss = loss.clone().detach().cpu().numpy()
		val_loss += epoch_loss
		_, predicted = out.max(1)
		total += targets.size(0)

		if log_preds:
			for el in targets:
				target_list.append( int(el.cpu().numpy()) )
			for el in predicted:
				pred_list.append( int(el.cpu().numpy()) )
			for el in out_dict['feats']:
				feat_list.append( el.flatten().cpu().numpy() )

		correct_batch = predicted.eq(targets).sum().item()
		correct += correct_batch

		if record_history:
			history.new_batch()
			history.record_batch('val_loss', epoch_loss)
			history.record_batch('val_acc', 100.0 * correct_batch / len(targets))
			history.record_batch('val_batch_size', len(targets))

	if log_preds:
		target_list = np.array(target_list)
		pred_list = np.array(pred_list)
		feat_list = np.array(feat_list)
		subject_list = dataloader_val.dataset.metadata_concat['subject'].values
		pred_dict = {'pred': pred_list,
					 'feat': feat_list,
					 'target': target_list,
					 'subject': subject_list
					 }
		acc_manual = 100 * np.sum(target_list == pred_list) / len(pred_list)
	else:
		acc_manual = 0.0
		pred_dict = {}

	return dataloader_val, net, criterion, history, args, acc_manual, pred_dict

def update_history_on_phase_end(history, phase, get_acc=False, acc=None):

	batch_history = history[-1]['batches']
	batch_sizes = np.array([el['{}_batch_size'.format(phase)] for el in batch_history if '{}_batch_size'.format(phase) in el.keys()])
	batch_losses = np.array([el['{}_loss'.format(phase)] for el in batch_history if '{}_loss'.format(phase) in el.keys()])
	batch_accs = np.array([el['{}_acc'.format(phase)] for el in batch_history if '{}_acc'.format(phase) in el.keys()])

	epoch_phase_loss = np.sum(batch_sizes * batch_losses) / np.sum(batch_sizes)
	if get_acc:
		epoch_phase_acc = acc
	else:
		epoch_phase_acc = np.sum(batch_sizes * batch_accs) / np.sum(batch_sizes)
	history.record('{}_loss'.format(phase), epoch_phase_loss)
	history.record('{}_acc'.format(phase), epoch_phase_acc)

	phase_losses = np.array([el['{}_loss'.format(phase)] for el in history[:]])
	phase_accs = np.array([el['{}_acc'.format(phase)] for el in history[:]])
	if epoch_phase_loss == np.min(phase_losses):
		history.record('{}_loss_best'.format(phase), True)
	if epoch_phase_acc == np.max(phase_accs):
		history.record('{}_acc_best'.format(phase), True)

	return history