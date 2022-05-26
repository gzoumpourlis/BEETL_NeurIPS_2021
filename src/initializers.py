import os
import sys
import pdb
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from src.config import *
from src.mne_utils import add_bipolar
from braindecode.models import *
from braindecode.datasets.base import *
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datasets.custom_dataset import CustomDataset
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.preprocess import (preprocess, Preprocessor)
from beetl.beetl_task_datasets import BeetlMILeaderboard, Cybathlon

import mne
mne.set_log_level('CRITICAL')

def init_dataset(args, dataset_name, cov_mix=False):

    print('Training on {} dataset'.format(dataset_name))

    # Using dataset details from src/config.py
    subject_range = get_subject_list(dataset_name)
    dataset_targets = get_target_list(dataset_name)
    dataset_events = get_event_list(dataset_name)

    print('\nLoading dataset')
    t1 = time.time()

    if dataset_name=='BCI_IV_2a':
        dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subject_range)
    elif dataset_name=='Cho2017':
        # subjects [32, 46, 49] removed from dataset
        # turned off log.warning
        dataset = MOABBDataset(dataset_name="Cho2017", subject_ids=subject_range)
    elif dataset_name=='PhysionetMI':
        # subjects [104, 106] removed from dataset, due to weird trial lengths
        dataset = MOABBDataset(dataset_name="PhysionetMI", subject_ids=subject_range)

    t2 = time.time()
    print('Time elapsed: {:.2f} seconds'.format(t2 - t1))

    # Preprocessing
    low_cut_hz = args.fmin  # low cut frequency for filtering
    high_cut_hz = args.fmax  # high cut frequency for filtering

    preprocessors = [
        Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
        Preprocessor('pick_channels', ch_names=mutual_channels_all, ordered=True), # pick electrodes, re-order
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    ]

    # Apply Common Average Reference
    if args.use_car:
        preprocessors.append(Preprocessor('set_eeg_reference', ref_channels='average', ch_type='eeg'))
    # Change frequency
    preprocessors.append(Preprocessor('resample', sfreq=args.sfreq))

    print('\nPreprocessing dataset')
    t1 = time.time()
    preprocess(dataset, preprocessors)
    t2 = time.time()
    print('Time elapsed: {:.2f} seconds'.format(t2-t1))

    ######################################################################
    # Window dataset

    # get window from tmin to tmax
    trial_start_offset_seconds = args.tmin
    trial_stop_offset_seconds = 0.0
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
    trial_stop_offset_samples = int(trial_stop_offset_seconds * sfreq)

    print('\nWindowing dataset')
    t1 = time.time()
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        preload=True,
    )
    t2 = time.time()
    print('Time elapsed: {:.2f} seconds'.format(t2 - t1))

    ######################################################################
    # Split dataset into train and valid.

    subject_column = windows_dataset.description['subject'].values
    subject_id = args.subject
    inds_train = list(np.where(subject_column != subject_id)[0])
    inds_val = list(np.where(subject_column == subject_id)[0])

    splitted = windows_dataset.split([inds_train, inds_val])
    train_set = splitted['0']
    val_set = splitted['1']

    # Merge multiple datasets into a single WindowDataset

    print('\nCreating custom dataset')
    t1 = time.time()

    dataset_train = CustomDataset(windows_dataset=train_set,
                                  phase='train',
                                  dataset_name=dataset_name,
                                  targets=dataset_targets,
                                  events=dataset_events,
                                  use_bipolar=args.use_bipolar,
                                  batch_align=args.batch_align,
                                  cov_mix=cov_mix
                                  )
    dataset_val = CustomDataset(windows_dataset=val_set,
                                phase='val',
                                dataset_name=dataset_name,
                                targets=dataset_targets,
                                events=dataset_events,
                                use_bipolar=args.use_bipolar,
                                batch_align=args.batch_align,
                                cov_mix=False
                                )

    # In case that "pre_align" instead of "batch_align" is done:
    # Align datasets only once, before starting the train/val process
    if args.batch_align==False:
        if args.pre_align:
            dataset_train.covariance_align()
            dataset_val.covariance_align()

    t2 = time.time()
    print('Time elapsed: {:.2f} seconds'.format(t2 - t1))

    # Create dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=True, num_workers=0)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size_val,
                                shuffle=False, pin_memory=True, num_workers=0)

    dataset_info = {'N_electrodes': dataset_train.N_electrodes,
                    'N_classes': dataset_train.N_classes,
                    'N_trials': dataset_train.N_trials,
                    'window_size': dataset_train.window_size
                    }
    print('\n{} dataset | Dataset info:'.format(dataset_name))
    print(dataset_info)

    return dataloader_train, dataloader_val, dataset_info

def init_BeetlMI_dataset(args, phase=None, shuffle=None, batch_size=None, cov_mix=False):

    # This function refers to the BeetlMI Leaderboard dataset of the 1st phase

    dataset_name = 'BeetlMI'
    subject_range = get_subject_list(dataset_name)
    dataset_targets = get_target_list(dataset_name)
    dataset_events = get_event_list(dataset_name)

    if batch_size==None:
        batch_size = args.batch_size

    low_cut_hz = args.fmin  # low cut frequency for filtering
    high_cut_hz = args.fmax  # high cut frequency for filtering

    #############################

    beetl_dataset = BeetlMILeaderboard()
    path = beetl_dataset.download()

    # X_train_A: (200, 63, 2000) | X_test_A: (400, 63, 2000)
    # X_train_b: (360, 32, 800)  | X_test_b: (600, 32, 800)

    ch_names_A = channels_BeetlMI_A
    ch_names_B = channels_BeetlMI_B

    sfreq_A = 500
    sfreq_B = 200
    trial_duration = 4.0

    # initialize list of datasets per subject
    ds_list = []

    for subject_id in subject_range:

        # When using BeetlMI Leaderboard dataset as a validation set (while training a model on Cybathlon),
        # we have chosen to discard subject #2, as its statistics (covariance) seem quite different from Cybathlon's
        if subject_id==2:
            continue

        description = {'subject': subject_id,
                       'session': phase,
                       'run': 0
                       }

        if subject_id < 3:
            if phase=='train':
                # Loading data (X) and labels (y) of labelled set
                X, y, _ = beetl_dataset.get_data(subjects=[subject_id, ], dataset='A')
            elif phase=='val':
                # Loading data (X) of unlabelled set
                _, _, X = beetl_dataset.get_data(subjects=[subject_id, ], dataset='A')
                # Creating fake labels for test set, as a vector filled with zeros.
                y = np.zeros(X.shape[0]).astype(np.uint8)
                # To overcome an MNE-related error coming up when adding bipolar channels to an Epochs object,
                # we had to include labels of all classes (i.e. 0,1,2,3) in the event IDs of the object.
                # Thus, we simply added one fake label for each one of the classes with event IDs 1, 2, and 3
                y[1] = 1
                y[2] = 2
                y[3] = 3
            sfreq = sfreq_A
            ch_names = ch_names_A

            # Amplitude is in Volts. Multiply to convert into uVolts.
            X = X * 1e+6
        else:
            if phase == 'train':
                # Loading data (X) and labels (y) of labelled set
                X, y, _ = beetl_dataset.get_data(subjects=[subject_id, ], dataset='B')
            elif phase == 'val':
                # Loading data (X) of unlabelled set
                _, _, X = beetl_dataset.get_data(subjects=[subject_id, ], dataset='B')
                # Creating fake labels for test set, as a vector filled with zeros.
                y = np.zeros(X.shape[0]).astype(np.uint8)
                # To overcome an MNE-related error coming up when adding bipolar channels to an Epochs object,
                # we had to include labels of all classes (i.e. 0,1,2,3) in the event IDs of the object.
                # Thus, we simply added one fake label for each one of the classes with event IDs 1, 2, and 3
                # This can be omitted, in case that we do not include bipolar channels
                y[1] = 1
                y[2] = 2
                y[3] = 3

            # Amplitude is in uVolts.

            sfreq = sfreq_B
            ch_names = ch_names_B

        #################################################################
        # Creating an array of events.
        events = []
        for k, v in enumerate(y):
            onset_time = int(100 + k * sfreq * trial_duration)
            events.append([onset_time, 0, v])
        events = np.asarray(events).astype(np.uint32)

        # Creating metadata
        metadata = pd.DataFrame({
            'i_window_in_trial': np.zeros(len(y)).astype(np.uint32),
            'i_start_in_trial': events[:, 0].squeeze().astype(np.uint32),
            'i_stop_in_trial': (events[:, 0].squeeze() + sfreq * trial_duration).astype(np.uint32),
            'target': y})

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg', verbose=None)

        # Creating Epochs object
        epochs = mne.EpochsArray(data=X, info=info, events=events, tmin=0, \
                                       event_id=dataset_events, reject=None, flat=None, reject_tmin=None, \
                                       reject_tmax=None, baseline=None, proj=True, on_missing='ignore', verbose=None)
        epochs.metadata = metadata

        #####################
        # Pick the channels that are common for BeetlMI and PhysionetMI
        epochs.pick_channels(mutual_channels_all, ordered=True)
        epochs_data = epochs.get_data()

        # Apply notch filter on 50Hz
        epochs_data = mne.filter.notch_filter(epochs_data, sfreq, freqs=[50,], \
                                                    filter_length='auto', notch_widths=None, trans_bandwidth=1, \
                                                    method='fir', iir_params=None, mt_bandwidth=None, \
                                                    p_value=0.05, picks=None, n_jobs=8)
        # Apply bandpass filter (fmin, fmax)
        epochs_data = mne.filter.filter_data(epochs_data, sfreq, low_cut_hz, high_cut_hz, picks=None, \
                                                   filter_length='auto', l_trans_bandwidth='auto',
                                                   h_trans_bandwidth='auto', n_jobs=8)

        # Recompute events array
        events = []
        for k, v in enumerate(y):
            onset_time = int(100 + k * sfreq * trial_duration)
            events.append([onset_time, 0, v])
        events = np.asarray(events).astype(np.uint32)

        info = mne.create_info(ch_names=mutual_channels_all, sfreq=sfreq, ch_types='eeg', verbose=None)
        # Create EpochsArray object
        epochs = mne.EpochsArray(data=epochs_data, info=info, events=events, tmin=0, \
                                       event_id=dataset_events, reject=None, flat=None, reject_tmin=None, \
                                       reject_tmax=None, baseline=None, proj=True, on_missing='ignore', verbose=None)
        # Apply Common Average Referencing (CAR)
        if args.use_car:
            epochs = epochs.set_eeg_reference()
        # Add bipolar channels (C3-C4 and C4-C3)
        if args.use_bipolar:
            epochs = add_bipolar(epochs)
        # Resample to have the same frequency across datasets
        epochs.resample(sfreq=args.sfreq)

        trial_duration_new = args.tmax - args.tmin
        # Assuming tmin is 0.0 on BeetlMI
        trial_duration_current = epochs.get_data().shape[-1]
        if ((args.sfreq*trial_duration_new)!=trial_duration_current):
            epochs.crop(tmin=args.tmin, tmax=args.tmax)

        events_new = []
        for k, v in enumerate(y):
            onset_time = int(100 + k * args.sfreq * trial_duration_new)
            events_new.append([onset_time, 0, v])
        events_new = np.asarray(events_new).astype(np.uint32)

        metadata_new = pd.DataFrame({
            'i_window_in_trial': np.zeros(len(y)).astype(np.uint32),
            'i_start_in_trial': events_new[:, 0].squeeze().astype(np.uint32),
            'i_stop_in_trial': (events_new[:, 0].squeeze() + args.sfreq * trial_duration_new).astype(np.uint32),
            'target': y})
        epochs.events = events_new
        epochs.metadata = metadata_new

        #####################

        ds = WindowsDataset(windows=epochs, description=description, transform=None)
        ds_list.append(ds)
    # Concatenate the datasets of all subjects
    windows_dataset = BaseConcatDataset(ds_list)

    ######################################################################
    # Merge multiple datasets into a single WindowDataset

    dataset = CustomDataset(windows_dataset=windows_dataset,
                            phase=phase,
                            dataset_name=dataset_name,
                            targets=dataset_targets,
                            events=dataset_events,
                            batch_align=args.batch_align,
                            cov_mix=cov_mix,
                            coef_cov_mix=args.coef_cov_mix
                            )

    # Align dataset using covariances
    if args.pre_align:
        dataset.covariance_align()

    # Get dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, pin_memory=True, num_workers=0)

    dataset_info = {'N_electrodes': dataset.N_electrodes,
                    'N_classes': dataset.N_classes,
                    'N_trials': dataset.N_trials,
                    'window_size': dataset.window_size
                    }
    print('\nBeetlMI Leaderboard | Dataset of phase: {} | Dataset info:'.format(phase))
    print(dataset_info)

    return dataloader, dataset_info

def init_Cybathlon_dataset(args, phase=None, shuffle=None, batch_size=None, cov_mix=False):

    # This function refers to the Cybathlon dataset of the 2nd phase

    dataset_name = 'Cybathlon'
    subject_range = get_subject_list(dataset_name)
    dataset_targets = get_target_list(dataset_name)
    dataset_events = get_event_list(dataset_name)

    if batch_size==None:
        batch_size = args.batch_size

    low_cut_hz = args.fmin  # low cut frequency for filtering
    high_cut_hz = args.fmax  # high cut frequency for filtering

    #############################
    # Write code to download dataset
    cybathlon_dataset = Cybathlon()

    ch_names_A = channels_Cybathlon_A
    ch_names_B = channels_Cybathlon_B

    sfreq_A = 500
    sfreq_B = 200
    trial_duration = 4.0

    # initialize list of datasets per subject
    ds_list = []

    # old_preds_filepath = os.path.join(os.getcwd(), 'beetl', 'gt.npy')
    # old_preds_dict = np.load(old_preds_filepath, allow_pickle=True)
    # old_preds_dict = old_preds_dict.item()
    # old_preds = old_preds_dict['pred']
    # old_preds_subject = old_preds_dict['subject']

    for subject_id in subject_range:

        description = {'subject': subject_id,
                       'session': phase,
                       'run': 0
                       }

        if subject_id < 4:
            if phase=='train':
                # Loading data (X) and labels (y) of labelled set
                X, y, _ = cybathlon_dataset.get_data(subjects=[subject_id, ], dataset='A')
            elif phase=='val':
                # Loading data (X) of unlabelled set
                _, _, X = cybathlon_dataset.get_data(subjects=[subject_id, ], dataset='A')

                # inds_subject = np.where(old_preds_subject==subject_id)[0]
                # y_subject = old_preds[inds_subject].astype(np.uint8)
                # y = y_subject.copy()

                # Creating fake labels for test set, as a vector filled with zeros.
                y = np.zeros(X.shape[0]).astype(np.uint8)
                # To overcome an MNE-related error coming up when adding bipolar channels to an Epochs object,
                # we had to include labels of all classes (i.e. 0,1,2,3) in the event IDs of the object.
                # Thus, we simply added one fake label for each one of the classes with event IDs 1, 2, and 3
                # This can be omitted, in case that we do not include bipolar channels
                y[1] = 1
                y[2] = 2
                y[3] = 3
            sfreq = sfreq_A
            ch_names = ch_names_A

            # Amplitude is in Volts. Multiply to convert into uVolts.
            X = X * 1e+6
        else:
            if phase == 'train':
                # Loading data (X) and labels (y) of labelled set
                X, y, _ = cybathlon_dataset.get_data(subjects=[subject_id, ], dataset='B')
            elif phase == 'val':
                # Loading data (X) of unlabelled set
                _, _, X = cybathlon_dataset.get_data(subjects=[subject_id, ], dataset='B')

                # inds_subject = np.where(old_preds_subject == subject_id)[0]
                # y_subject = old_preds[inds_subject].astype(np.uint8)
                # y = y_subject.copy()

                # Creating fake labels for test set, as a vector filled with zeros.
                y = np.zeros(X.shape[0]).astype(np.uint8)
                # To overcome an MNE-related error coming up when adding bipolar channels to an Epochs object,
                # we had to include labels of all classes (i.e. 0,1,2,3) in the event IDs of the object.
                # Thus, we simply added one fake label for each one of the classes with event IDs 1, 2, and 3
                # This can be omitted, in case that we do not include bipolar channels
                y[1] = 1
                y[2] = 2
                y[3] = 3

            # Amplitude is in Volts.

            sfreq = sfreq_B
            ch_names = ch_names_B

        #################################################################
        # Creating an array of events.
        events = []
        for k, v in enumerate(y):
            onset_time = int(100 + k * sfreq * trial_duration)
            events.append([onset_time, 0, v])
        events = np.asarray(events).astype(np.uint32)

        # Creating metadata
        metadata = pd.DataFrame({
            'i_window_in_trial': np.zeros(len(y)).astype(np.uint32),
            'i_start_in_trial': events[:, 0].squeeze().astype(np.uint32),
            'i_stop_in_trial': (events[:, 0].squeeze() + sfreq * trial_duration).astype(np.uint32),
            'target': y})

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg', verbose=None)

        # Creating Epochs object
        epochs = mne.EpochsArray(data=X, info=info, events=events, tmin=0, \
                                       event_id=dataset_events, reject=None, flat=None, reject_tmin=None, \
                                       reject_tmax=None, baseline=None, proj=True, on_missing='ignore', verbose=None)
        epochs.metadata = metadata

        #####################
        # Pick the channels that are common for Cybathlon and PhysionetMI
        epochs.pick_channels(mutual_channels_all, ordered=True)
        epochs_data = epochs.get_data()

        # Apply notch filter on 50Hz
        epochs_data = mne.filter.notch_filter(epochs_data, sfreq, freqs=[50,], \
                                                    filter_length='auto', notch_widths=None, trans_bandwidth=1, \
                                                    method='fir', iir_params=None, mt_bandwidth=None, \
                                                    p_value=0.05, picks=None, n_jobs=8)
        # Apply bandpass filter (fmin, fmax)
        epochs_data = mne.filter.filter_data(epochs_data, sfreq, low_cut_hz, high_cut_hz, picks=None, \
                                                   filter_length='auto', l_trans_bandwidth='auto',
                                                   h_trans_bandwidth='auto', n_jobs=8)

        # Recompute events array
        events = []
        for k, v in enumerate(y):
            onset_time = int(100 + k * sfreq * trial_duration)
            events.append([onset_time, 0, v])
        events = np.asarray(events).astype(np.uint32)

        info = mne.create_info(ch_names=mutual_channels_all, sfreq=sfreq, ch_types='eeg', verbose=None)
        # Create EpochsArray object
        epochs = mne.EpochsArray(data=epochs_data, info=info, events=events, tmin=0, \
                                       event_id=dataset_events, reject=None, flat=None, reject_tmin=None, \
                                       reject_tmax=None, baseline=None, proj=True, on_missing='ignore', verbose=None)
        # Apply Common Average Referencing (CAR)
        if args.use_car:
            epochs = epochs.set_eeg_reference()
        # Add bipolar channels (C3-C4 and C4-C3)
        if args.use_bipolar:
            epochs = add_bipolar(epochs)
        # Resample to have the same frequency across datasets
        epochs.resample(sfreq=args.sfreq)

        trial_duration_new = args.tmax - args.tmin
        # Assuming tmin is 0.0 on BeetlMI
        trial_duration_current = epochs.get_data().shape[-1]
        if ((args.sfreq*trial_duration_new)!=trial_duration_current):
            epochs.crop(tmin=args.tmin, tmax=args.tmax)

        events_new = []
        for k, v in enumerate(y):
            onset_time = int(100 + k * args.sfreq * trial_duration_new)
            events_new.append([onset_time, 0, v])
        events_new = np.asarray(events_new).astype(np.uint32)

        metadata_new = pd.DataFrame({
            'i_window_in_trial': np.zeros(len(y)).astype(np.uint32),
            'i_start_in_trial': events_new[:, 0].squeeze().astype(np.uint32),
            'i_stop_in_trial': (events_new[:, 0].squeeze() + args.sfreq * trial_duration_new).astype(np.uint32),
            'target': y})
        epochs.events = events_new
        epochs.metadata = metadata_new

        #####################

        ds = WindowsDataset(windows=epochs, description=description, transform=None)
        ds_list.append(ds)
    # Concatenate the datasets of all subjects
    windows_dataset = BaseConcatDataset(ds_list)

    ######################################################################
    # Merge multiple datasets into a single WindowDataset

    dataset = CustomDataset(windows_dataset=windows_dataset,
                            phase=phase,
                            dataset_name=dataset_name,
                            targets=dataset_targets,
                            events=dataset_events,
                            batch_align=args.batch_align,
                            cov_mix=cov_mix,
                            coef_cov_mix=args.coef_cov_mix
                            )

    # Align dataset using covariances
    if args.pre_align:
        dataset.covariance_align()

    # Get dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, pin_memory=True, num_workers=0)

    dataset_info = {'N_electrodes': dataset.N_electrodes,
                    'N_classes': dataset.N_classes,
                    'N_trials': dataset.N_trials,
                    'window_size': dataset.window_size
                    }
    print('\nCybathlon | Dataset of phase: {} | Dataset info:'.format(phase))
    print(dataset_info)

    return dataloader, dataset_info

def init_net(args, dataset_info):

    n_classes = dataset_info['N_classes']
    n_chans = dataset_info['N_electrodes']

    input_window_samples = dataset_info['window_size']

    if args.net_cls=='eegnetv4':
        net = EEGNetv4_class(n_chans,
                            n_classes,
                            input_window_samples=input_window_samples,
                            final_conv_length="auto",
                            pool_mode="max",
                            F1=8,
                            D=2,
                            F2=16,
                            kernel_length=64,
                            drop_prob=args.dropout,
                            finetune=args.finetune
        )
    else:
        print('Network model not implemented yet. Quitting...')
        quit()

    if args.cuda:
        net.cuda()

    return net

def init_discr(args):

    if args.da_type=='dann':
        discr = discriminator_DANN(n_inputs=192)
    else:
        print('Unknown type of Domain Adaptation. Could not initialize discriminator network. Quitting...')
        quit()

    if args.cuda:
        discr.cuda()

    return discr

def init_criterion(args):

    criterion = nn.CrossEntropyLoss()

    return criterion

def init_optimizer(args, net, discr=None):

    if discr is None:
        parameters = net.parameters()
    else:
        parameters = list(net.parameters()) + list(discr.parameters())

    if args.optim == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

    return optimizer