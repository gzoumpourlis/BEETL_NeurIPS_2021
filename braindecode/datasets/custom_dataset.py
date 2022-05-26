import os
import sys
import mne
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.geodesic import geodesic_riemann
from scipy.linalg import pinv, sqrtm

sys.path.insert(0, os.getcwd())
from src.mne_utils import add_bipolar

def intersection(list1, list2):

    # find common elements from two lists

    list3 = [value for value in list1 if value in list2]

    return list3

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        eeg, target = sample['eeg'], sample['target']
        sample_tensor = {'eeg': torch.from_numpy(eeg),
                        'target': torch.from_numpy(np.array(target)).type(torch.LongTensor),
                         }

        return sample_tensor

class CustomDataset(Dataset):

    def __init__(self, windows_dataset,
                 phase=None,
                 dataset_name=None,
                 targets=None,
                 events=None,
                 use_bipolar=False,
                 batch_align=False,
                 cov_mix=False,
                 coef_cov_mix=0.0
                 ):
        """
        Args:
            phase (string): The phase (training, validation/test) that the dataset is used.
            dataset_name (string): Dataset name
            targets (list): List containing the names of the target classes
            events (dictionary): Dictionary with the correspondence class_name-class_ID
            use_bipolar (boolean): If True, two bipolar channels are added to the EEG data
            batch_align (boolean): If True, then EEG aligning is done on-the-fly for each batch
            cov_mix (boolean): If True, then the covariance matrices of two sessions (one labelled, one unlabelled),
                                are mixed to align the EEG data of the labelled session, considering the
                                statistics of the unlabelled session.
        """
        self.targets = targets
        self.N_classes = len(self.targets)
        self.targets_rearranged = [el for el in range(self.N_classes)]

        self.phase = phase
        self.dataset_name = dataset_name
        self.use_bipolar = use_bipolar
        self.batch_align = batch_align
        self.cov_mix = cov_mix
        self.coef_cov_mix = coef_cov_mix

        self.events_rearranged = {}
        self.label_correspondence = {}
        for target_name, target_num in zip(self.targets, self.targets_rearranged):
            self.events_rearranged.update( {target_name: target_num} )
            self.label_correspondence.update({events[target_name]: target_num}) # old_label --> new_label

        # print('Events rearranged: {}'.format(self.events_rearranged))
        # print('label_correspondence: {}'.format(self.label_correspondence))

        # Subsample events, using only the kept targets (when discarding some classes from a dataset)
        self.events = {}
        self.event_ids = []
        for target in self.targets:
            self.events.update( {target: events[target]} )
            self.event_ids.append(events[target])

        ###################################################

        list_subject = []
        list_session = []
        list_run = []
        list_metadata = []
        for ds in windows_dataset.datasets:
            sub = ds.description['subject']
            sess = ds.description['session']
            run = ds.description['run']
            metadata = ds.windows.metadata
            list_subject.append(sub)
            list_session.append(sess)
            list_run.append(run)
            list_metadata.append(metadata)

        ###################################################
        # Merge runs per session

        list_merged_subject = []
        list_merged_session = []
        list_merged_metadata = []
        list_merged_epochs = []
        list_merged_ntrials = []

        # This list will be used to uniquely identify each sample of the dataset, with a pair of indices:
        # sub-dataset index, and trial index
        list_sample_indices = []

        unique_subjects = []
        unique_sessions = []
        for sub in list_subject:
            if sub not in unique_subjects:
                unique_subjects.append(sub)
        for sess in list_session:
            if sess not in unique_sessions:
                unique_sessions.append(sess)

        cnt_trials = 0
        dataset_index = 0
        for sub in unique_subjects:
            indices_sub = [i for i, x in enumerate(list_subject) if x == sub]
            for sess in unique_sessions:
                indices_sess = [i for i, x in enumerate(list_session) if x == sess]
                # sub/sess intersection
                indices_sub_sess = intersection(indices_sub, indices_sess)

                if not indices_sub_sess:
                    # empty list
                    continue

                # Concatenate metadata and epochs from multiple runs
                metadata = pd.concat([windows_dataset.datasets[index].windows.metadata for index in indices_sub_sess])
                epochs = mne.concatenate_epochs([windows_dataset.datasets[index].windows for index in indices_sub_sess])

                # Add bipolar channels (C3-C4 and C4-C3)
                if self.use_bipolar:
                    # For BeetlMI_Leaderboard and Cybathlon datasets, the bipolar channels are added in this point
                    # For PhysionetMI dataset, the bipolar channels are added at another point,
                    # with the pipeline remaining entirely the same
                    if (self.dataset_name != 'BeetlMI') and (self.dataset_name != 'Cybathlon'):
                        epochs = add_bipolar(epochs)

                subject_column = [sub for row_cnt in range(len(metadata))]
                metadata['subject'] = subject_column

                ####################################
                # Subsample trials, belonging in a subset of classes (picking *some* samples per class)
                # This is done to ensure class balance, by undersampling classes that have many samples
                # Subsampling is done on Physionet. It is *not* done on BeetlMI and Cybathlon datasets
                if (self.dataset_name != 'BeetlMI') and (self.dataset_name != 'Cybathlon'):
                    trials_per_class = list()
                    for event_id, target in zip(self.event_ids, self.targets):
                        trials_per_class.append(len(epochs[target]))
                    # number of samples for each class
                    trials_per_class = np.array(trials_per_class)
                    # cardinality of each class after subsampling
                    min_n_trials_per_class = np.min(trials_per_class)

                    vec_targets = metadata['target'].values
                    condition_array = np.full((len(vec_targets)), False, dtype=bool)
                    for event_id in self.event_ids:
                        inds_event = np.where(vec_targets == event_id)[0]
                        condition_event = np.full((len(vec_targets)), False, dtype=bool)
                        inds_event_kept = random.sample(list(inds_event), min_n_trials_per_class)
                        condition_event[inds_event_kept] = True
                        condition_array = condition_array | condition_event
                    # indices of samples that will be kept
                    inds_event_kept_all = np.where(condition_array==True)[0]

                    # updating the metadata, according to the subsampling
                    metadata = metadata[condition_array]
                    # EEG data subsampling is done here
                    epochs = epochs[inds_event_kept_all]

                ####################################
                epochs = epochs.get_data()
                n_trials = epochs.shape[0]

                # Filling the list to uniquely identify samples. This will be used in the dataloader
                for trial_index in range(n_trials):
                    list_sample_indices.append([dataset_index, trial_index])

                list_merged_ntrials.append(n_trials)
                cnt_trials += n_trials
                dataset_index += 1

                list_merged_subject.append(sub)
                list_merged_session.append(sess)
                list_merged_metadata.append(metadata)
                list_merged_epochs.append(epochs)

        self.subject = list_merged_subject
        self.session = list_merged_session
        self.ntrials = list_merged_ntrials
        self.metadata = list_merged_metadata
        self.epochs = list_merged_epochs

        self.metadata_concat = pd.concat(self.metadata)
        # number of sub-datasets
        self.N_datasets = len(self.subject)
        # total number of trials for the whole dataset
        self.N_trials = cnt_trials
        # final list of samples (their indices)
        self.sample_indices = list_sample_indices
        # number of EEG channels (in case of added bipolar channels, this does not correspond exactly to #Electrodes)
        self.N_electrodes = self.epochs[0].shape[1]
        # temporal length of EEG epochs
        self.window_size = self.epochs[0].shape[2]

        # Compute trial-wise and session-wise covariances for each sub-dataset
        self.compute_covariances()

        self.transform = ToTensor()

    def compute_covariances(self):

        self.trialwise_covs_per_dataset = list()
        self.dataset_covs = list()

        for i in range(self.N_datasets):
            # data of a session
            data = self.epochs[i]
            trialwise_covs_list = list()
            # Compute covariances using pyriemann package, to avoid errors when having linearly dependent channels
            trial_covs = Covariances('oas').fit_transform(data)
            for i_window in range(data.shape[0]):
                # trial-wise covariance
                trial_cov = trial_covs[i_window]
                trialwise_covs_list.append(trial_cov)
            trialwise_covs = np.array(trialwise_covs_list)
            self.trialwise_covs_per_dataset.append(trialwise_covs)
            # session-wise covariance is the Riemannian mean of the trial-wise covariances of the session
            dataset_cov = mean_riemann(trialwise_covs, tol=1e-08, maxiter=50, init=None, sample_weight=None)
            self.dataset_covs.append(dataset_cov)

        self.dataset_covs = np.array(self.dataset_covs)
        self.trialwise_covs_per_dataset = np.array(self.trialwise_covs_per_dataset, dtype=object)

        ########################
        # Compute projection matrix inv(sqrtm(cov)) [i.e. cov^(-1/2)], that is used to align the EEG data

        projs_list = list()
        for i in range(self.N_datasets):
            dataset_cov = self.dataset_covs[i]
            proj = pinv(sqrtm(dataset_cov)).real
            projs_list.append(proj)
        projs = np.array(projs_list)
        self.projs = projs

    def covariance_align(self):

        # Align the EEG data, using the projection matrix computed based on the covariance matrices
        # This function is used to align the EEG data only once, in an offline manner,
        # i.e. before starting the training process

        for i in range(self.N_datasets):
            proj = pinv(sqrtm(self.dataset_covs[i])).real
            for i_window in range(self.ntrials[i]):
                data_epoch = self.epochs[i][i_window]
                aligned_data_epoch = np.matmul(proj, data_epoch)
                self.epochs[i][i_window] = aligned_data_epoch

    def __len__(self):
        return self.N_trials

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        dataset_index = self.sample_indices[idx][0]
        trial_index = self.sample_indices[idx][1]

        # Getting data of the sample
        x = self.epochs[dataset_index][trial_index].copy()
        # Getting label of the sample
        y = self.metadata[dataset_index]['target'].values[trial_index]

        # Mapping labels from all datasets, to have the same class-value correspondence
        # print('Changing label| {} --> {}'.format(y, self.label_correspondence[y] ))
        y = self.label_correspondence[y]

        if self.batch_align:
            if self.cov_mix:
                # covariance matrix of labelled session
                cov_train = self.dataset_covs[dataset_index].copy()
                # covariance matrix of unlabelled session
                cov_val = self.dataset_unlabelled_covs[dataset_index].copy()
                # mixing the two covariance matrices, in a Euclidean manner
                cov_mixed = (1 - self.coef_cov_mix)*cov_train + self.coef_cov_mix*cov_val
                # could be done in a Riemannian manner
                # cov_mixed = geodesic_riemann(cov_train, cov_val, alpha=self.coef_cov_mix)
                # computing matrix cov_mixed^(-1/2), that will be used to align the data of the labelled session
                proj = pinv(sqrtm(cov_mixed)).real
                # align data
                x = np.matmul(proj, x)
            else:
                # obtaining precomputed matrix cov^(-1/2), that can be used to align the data of any session
                proj = self.projs[dataset_index].copy()
                # align data
                x = np.matmul(proj, x)

        sample = {'eeg': x, 'target': y}

        if self.transform:
            sample = self.transform(sample)

        return sample
