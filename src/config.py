def get_subject_list(dataset_name):

    if dataset_name=='BCI_IV_2a':
        subject_list = subjects_BCI_IV_2a
    elif dataset_name=='Cho2017':
        subject_list = subjects_Cho2017
    elif dataset_name=='PhysionetMI':
        subject_list = subjects_PhysionetMI
    elif dataset_name=='BeetlMI':
        subject_list = subjects_BeetlMI
    elif dataset_name=='Cybathlon':
        subject_list = subjects_Cybathlon

    return subject_list

def get_target_list(dataset_name):

    if dataset_name=='BCI_IV_2a':
        target_list = targets_BCI_IV_2a
    elif dataset_name=='Cho2017':
        target_list = targets_Cho2017
    elif dataset_name=='PhysionetMI':
        target_list = targets_PhysionetMI
    elif dataset_name=='BeetlMI':
        target_list = targets_BeetlMI
    elif dataset_name=='Cybathlon':
        target_list = targets_Cybathlon

    return target_list

def get_event_list(dataset_name):

    if dataset_name=='BCI_IV_2a':
        event_list = events_BCI_IV_2a
    elif dataset_name=='Cho2017':
        event_list = events_Cho2017
    elif dataset_name=='PhysionetMI':
        event_list = events_PhysionetMI
    elif dataset_name=='BeetlMI':
        event_list = events_BeetlMI
    elif dataset_name=='Cybathlon':
        event_list = events_Cybathlon

    return event_list
###############################################
# Subjects

subjects_BCI_IV_2a = [x for x in range(1, 10)] # 9
subjects_Cho2017 = [x for x in range(1, 53) if x not in [32, 46, 49]] # 52
subjects_PhysionetMI = [x for x in range(1, 110) if (x!=104 and x!=106)] # 109
subjects_BeetlMI = [x for x in range(1, 6)] # 5
subjects_Cybathlon = [x for x in range(1, 6)] # 5

###############################################
# Targets

# Kept targets
targets_BCI_IV_2a = ['left_hand', 'right_hand', 'feet', 'tongue']
targets_Cho2017 = ['left_hand', 'right_hand']
targets_PhysionetMI = ['left_hand', 'right_hand', 'feet', 'rest']
targets_BeetlMI = ['left_hand', 'right_hand', 'feet', 'rest']
targets_Cybathlon = ['left_hand', 'right_hand', 'feet', 'rest']

# Full targets
# targets_BCI_IV_2a = ['feet', 'left_hand', 'right_hand', 'tongue']
# targets_Cho2017 = ['left_hand', 'right_hand']
# targets_PhysionetMI = ['left_hand', 'rest', 'right_hand', 'feet', 'hands']

###############################################
# Events

events_BCI_IV_2a = {'feet': 0, 'left_hand': 1, 'right_hand': 2, 'tongue': 3}
events_Cho2017 = {'left_hand': 0, 'right_hand': 1}
events_PhysionetMI = {'left_hand': 0, 'rest': 1, 'right_hand': 2, 'feet': 3, 'hands': 4}
events_BeetlMI = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'rest': 3}
events_Cybathlon = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'rest': 3}

###############################################
# Electrodes

channels_BCI_IV_2a = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1",\
                     "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]

channels_Cho2017 = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7',\
                    'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz',\
                    'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',\
                    'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8',\
                    'P10', 'PO8', 'PO4', 'O2', 'EMG1', 'EMG2', 'EMG3', 'EMG4', 'Stim']

channels_PhysionetMI = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',\
                        'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz',\
                        'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8',\
                        'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3',\
                        'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz']

channels_BeetlMI_A = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', \
                  'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', \
                  'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', \
                  'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', \
                  'F2', 'AF4', 'AF8']

channels_BeetlMI_B = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',\
                      'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',\
                      'P6', 'P8']

channels_Cybathlon_A = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3',\
                        'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2',\
                        'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7',\
                        'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6',\
                        'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']

channels_Cybathlon_B = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2','FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',\
                        'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2',\
                        'P4', 'P6', 'P8']

###############################################
# Mutual electrodes

# Mutual Cho2017 & BCI_IV_2a: 22 channels
mutual_channels_Cho2017_BCI_IV_2a = ['FC3', 'FC1', 'C1', 'C3', 'C5', 'CP3', 'CP1', 'P1', 'POz', 'Pz', 'CPz', 'Fz',\
                                     'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'CP4', 'CP2', 'P2']

# Mutual PhysionetMI & BCI_IV_2a: 22 channels
mutual_channels_PhysionetMI_BCI_IV_2a = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',\
                                         'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'Fz', 'P1', 'Pz', 'P2', 'POz']

# Mutual BeetlMI_A & BCI_IV_2a: 21 channels
mutual_channels_BeetlMI_A_BCI_IV_2a = ['Fz', 'FC1', 'C3', 'CP1', 'Pz', 'CP2', 'C4', 'FC2', 'FC3', 'FCz', 'C1', 'C5',\
                                     'CP3', 'P1', 'POz', 'P2', 'CPz', 'CP4', 'C6', 'C2', 'FC4']

# Mutual BeetlMI_B & BCI_IV_2a: 18 channels
mutual_channels_BeetlMI_B_BCI_IV_2a = ['Fz', 'FC1', 'FC2', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1',\
                                       'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2']

# Mutual BeetlMI_A & BeetlMI_B: 31 channels
mutual_channels_BeetlMI_A_BeetlMI_B = ['FC1', 'Pz', 'F3', 'CP6', 'P5', 'CP5', 'CP1', 'CP3', 'Fz', 'FC6', 'P2', 'P4',\
                                       'FC5', 'P3', 'C5', 'CP4', 'P7', 'C1', 'C3', 'C4', 'P8', 'C2', 'F4', 'CPz',\
                                       'C6', 'FC2', 'P1', 'Fp2', 'CP2', 'Fp1', 'P6']

# Mutual BeetlMI_A & BeetlMI_B & BCI_IV_2a: 17 channels
mutual_channels_BeetlMI_BCI_IV_2a = ['C1', 'C4', 'P2', 'Pz', 'Fz', 'C3', 'CP1', 'CP2', 'CP4', 'C5', 'CPz', 'C6',\
                                     'FC2', 'P1', 'FC1', 'C2', 'CP3']

# Mutual BeetlMI & PhysionetMI: 31 channels
mutual_channels_BeetlMI_PhysionetMI = ['FC1', 'Pz', 'F3', 'CP6', 'P5', 'CP5', 'CP1', 'CP3', 'Fz', 'FC6', 'P2', 'P4',\
                                       'FC5', 'P3', 'C5', 'CP4', 'P7', 'C1', 'C3', 'C4', 'P8', 'C2', 'F4', 'CPz',\
                                       'C6', 'FC2', 'P1', 'Fp2', 'CP2', 'Fp1', 'P6']

mutual_channels_all = mutual_channels_BeetlMI_PhysionetMI