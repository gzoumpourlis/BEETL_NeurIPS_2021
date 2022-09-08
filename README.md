# Submission in BEETL Competition, NeurIPS 2021

## Overview

In this work, we present our method on the **Motor Imagery** classification task of the BEETL competition. <br />
Our method includes two stages of training: <br />
-  **1st stage: Pre-training on PhysionetMI dataset + Unsupervised Domain Adaptation on Cybathlon dataset** <br />
-  **2nd stage: Finetuning + Unsupervised Domain Adaptation + Covariance Adaptation on Cybathlon dataset** <br />

## Datasets

- We use the [**PhysionetMI**](https://archive.physionet.org/pn4/eegmmidb/) dataset as our only external dataset, keeping the classes _left_hand_, _right_hand_, _feet_ and _rest_. <br />
- We use the [**BeetlMI Leaderboard**](https://figshare.com/articles/dataset/leaderboardMI/14839650) dataset to experiment in the development phase <br />
- We use the [**Cybathlon**](https://figshare.com/articles/dataset/finalMI/16586213) dataset to experiment in the final testing phase <br />

All datasets should be downloaded at the directory `~/mne_data`, with the following sub-directories:

```
~/mne_data/
├── MNE-eegbci-data/              (note: PhysionetMI)
├── MNE-beetlmileaderboard-data/  (note: BeetlMI)
└── MNE-beetlmitest-data/         (note: Cybathlon)
```

To download the Cybathlon dataset, you can execute the following commands

```bash
$  cd beetl
$  sh download_Cybathlon.sh	
```

Notes: <br />
- We discard the EEG recording of subjects #104 and #106 on PhysionetMI, due to inconsistencies on the trial lengths. <br />
- We keep the following 31 EEG electrodes, that exist in all of the datasets that we use: <br />
``
'FC1', 'Pz', 'F3', 'CP6', 'P5', 'CP5', 'CP1', 'CP3', 'Fz', 'FC6', 'P2', 'P4', 'FC5', 'P3', 'C5', 'CP4', 'P7', 'C1', 'C3', 'C4', 'P8', 'C2', 'F4', 'CPz', 'C6', 'FC2', 'P1', 'Fp2', 'CP2', 'Fp1', 'P6'
``
<br />
EEG preprocessing has the following steps: <br />
1) Bringing the EEG signals into the measurement unit of uV (microvolts) <br />
2) Notch filtering to remove the 50Hz component, when necessary <br /> 
3) Bandpass filtering in the range 4-38 Hz  <br />
4) Re-referencing the signals to the common average  <br />
5) Resampling signals to 100Hz  <br />
6) Adding two bipolar channels (specifically, the first bipolar channel is C3-C4 and the second bipolar channel is C4-C3). Thus, we have 33 EEG channels in total.  <br />

## Network Architectures

- **Motor imagery classification**: The network architecture that is used for motor imagery classification, is a variant of the [EEGNet implementation](https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegnet.py#L25) that exists in the [braindecode](https://github.com/braindecode/braindecode) toolbox.
The original paper of EEGNet can be found [here](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/)

- **Unsupervised Domain Adaptation**: The network that is used to perform unsupervised domain adaptation, is based on the well-known [DANN](https://proceedings.mlr.press/v37/ganin15.html) method, where a domain discriminator (classifier) is used, to predict whether a sample belongs to the source or target domain, combined with a Gradient Reversal Layer (GRL). The GRL implementation is the one that exists in the [python-domain-adaptation](https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py#L26) toolbox of @jvanvugt.

## Installation

We recommend installing the required packages using python's native virtual environment. For Python 3.4+, this can be done as follows:

```bash
$ python -m venv beetl_venv
$ source beetl_venv/bin/activate
(beetl_venv) $ pip install --upgrade pip
(beetl_venv) $ pip install -r requirements.txt
```


## Training (1st stage)

To train a model, you need to use `src/run_exp_train.py`, running the following command:

```bash
python src/run_exp_train.py
```

You can inspect the various arguments that are passed to `src/train.py`.

By default, the `src/run_exp_train.py` script performs: <br />
- **Training on PhysionetMI** dataset, as a first step that is later followed by finetuning on Cybathlon dataset. Batch size is set to `64` and the training will be conducted for `100` epochs. <br />
- **Unsupervised Domain Adaptation (UDA) on Cybathlon** dataset. UDA is done with an increasing contribution of the domain classification loss to the total optimization loss, from `0.0` to `0.3`, as the training process progresses. For details on the domain loss coefficient, check the original DANN paper, and the corresponding code at `src/train.py`. <br />

The typical cross-entropy loss is used as the criterion for motor imagery classification, where the targets are the `4` aforementioned classes of PhysionetMI, by undersampling occasionaly as needed, to keep the dataset balanced with respect to sample occurence per class.

SGD is selected as the optimizer (momentum=`0.9`, weight decay=`5e-4`), with a warmup period of `20` epochs where the learning rate is increased linearly from `1e-5` to `0.01`. After the initial warmup period, a cosine annealing scheduler is used to decrease the learning rate for the remaining `80` epochs. For this 1st stage, we use dropout with probability equal to `0.1` in EEGNet.

The Cybathlon training (labelled) set is used as the **validation set** of this stage. We can do that, as the labels of Cybathlon's training set have not been used yet, thus we get an idea of how our model generalizes on Cybathlon. The accuracy on this validation set is used to determine the best model, and save its weights in a **checkpoint**.

The training process will create the following directories:

```
json        (note: contains the arguments of each experiment, stored in JSON format)
checkpoints (note: contains the saved model checkpoints, stored as .pth files)
preds       (note: contains the predictions on a dataset, as a .txt file and as a .npy file)
```

<p>
  <img src="https://raw.githubusercontent.com/gzoumpourlis/BEETL_NeurIPS_2021/main/figures/stage_1.png" width="600" title="Overview of the 1st stage of training">
</p>

## Finetuning (2nd stage)

To finetune a model, you need to use `src/run_exp_finetune.py`, running the following command:

```bash
python src/run_exp_finetune.py
```

You can inspect the various arguments that are passed to `src/train.py`.

By default, the `src/run_exp_finetune.py` script performs: <br />
- **Finetuning on Cybathlon** dataset, specifically on its labelled (training) set. <br />
- **UDA on Cybathlon** dataset, specifically on its unlabelled (testing) set. UDA is done with a fixed contribution of the domain classification loss to the total optimization loss, as the domain loss coefficient is kept equal to `1.0`. <br />
- **Covariance Adaptation (CA) on Cybathlon** dataset, specifically on its labelled (training) set. CA adapts the covariance matrix of each participant in the training set (i.e. the Cybathlon labelled set in this case), taking into consideration the covariance matrix of the same participant on the test set (i.e. the Cybathlon unlabelled set in this case). This is performed in the `braindecode/datasets/custom_dataset.py` script. Covariance Adaptation is enabled by setting the `batch_align` argument to `True` (and `pre_align` argument to `False`). We set the covariance mixing coefficient equal to `0.5`. <br />

A pretrained model's checkpoint is loaded on EEGNet. The domain discriminator network is randomly initialized.

Batch size is set to `64` and finetuning is conducted for `130` epochs. The typical cross-entropy loss is used as the criterion for motor imagery classification, where the targets are the 4 aforementioned classes. No undersampling is applied in this stage.

AdamW is selected as the optimizer (weight decay=`5e-4`), using its [PyTorch implementation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) with a constant learning rate of `1e-4` (no warmup period, no scheduler). For more details on AdamW, you can check the original paper [here](https://openreview.net/forum?id=Bkg6RiCqY7).

For this 2nd stage, we use dropout with probability equal to `0.3` in EEGNet. We freeze the parameters of the first convolutional layer (temporal convolution) of EEGNet. We also insert an L2 normalization layer, between the last convolution layer and the MLP classifier of EEGNet during the 2nd stage.

The BeetlMI training (labelled) set is used as the **validation set** of this stage. We can do that, as the labels of BeetlMI's training set have not been used yet. We get an idea of how our model generalizes on BeetlMI, which is somehow similar to Cybathlon. We performed a visual inspection of the Riemannian distances between the covariance matrices for all the participants of BeetlMI and Cybathlon, labelled and unlabelled sets. The distance matrix can be found in image format, at the file `plots/cov_distances_BeetlMI_Cybathlon.png`. We decided to discard participant #2 from the BeetlMI training set, when using it for validation purposes to assume the generalization on Cybathlon, as it seemed to have the most dissimilar statistics. The accuracy on this validation set is used to determine the best model, and save its weights in a **checkpoint**. This checkpoint is used to obtain the predictions on Cybathlon test (unlabelled) set. Please note that to conform with the target classes as needed to evaluate a submission on the BEETL competition (i.e., merging the classes "_feet_" and "_rest_" into one class named "_other_"), we replace every predicted label of class ID `3` (i.e. 4th class using zero-indexing), to class ID `2`: _other_ (`0`: _left_hand_, `1`: _right_hand_).

<p>
  <img src="https://raw.githubusercontent.com/gzoumpourlis/BEETL_NeurIPS_2021/main/figures/stage_2.png" width="600" title="Overview of the 2nd stage of training">
</p>

## Model checkpoints

The models checkpoints are as follows:

```
./checkpoints/
├── net_best_pretrained.pth
└── net_best_finetuned.pth

```
- Checkpoint after the 1st (*pretraining*) stage: `checkpoints/net_best_pretrained.pth`
- Checkpoint after the 2nd (*finetuning*) stage: `checkpoints/net_best_finetuned.pth`

## Evaluation

To evaluate a model, you need to use `src/run_eval_pretrained.py` or `src/run_eval_finetuned.py`, running the following commands:

```bash
python src/run_eval_pretrained.py
```

or

```bash
python src/run_eval_finetuned.py
```

You can inspect the various arguments that are passed to `src/eval.py`.

The `src/run_eval_pretrained.py` script performs **evaluation on Cybathlon** dataset, specifically on its **labelled (training) set**. The obtained accuracy is **~40-41% on the 4-class problem**, i.e. when not merging the classes "_feet_" and "_rest_" into one class named "_other_". The checkpoint used for this, is available at the file `checkpoints/net_best_pretrained.pth`.

The `src/run_eval_finetuned.py` script performs **evaluation on Cybathlon** dataset, specifically on its **unlabelled (test) set**. The obtained accuracy, when submitting the results on [CodaLab's page for the final scoring phase of motor imagery](https://competitions.codalab.org/competitions/33427#participate-submit_results), is **~55.7% on the 3-class problem**, i.e. when merging the classes "_feet_" and "_rest_" into one class named "_other_". The checkpoint used for this, is available at the file `checkpoints/net_best_finetuned.pth`. _Please note that the accuracy printed on the terminal when running this script, does not have any meaning, as we load a vector full of zeros to be used as the groundtruth labels of Cybathlon's test set._



## Acknowledgement

The research of Georgios Zoumpourlis was supported by QMUL Principal's Studentship.

## Credits

The current GitHub repo contains code parts from the following repositories (sometimes heavily chopped, keeping only the necessary tools).
Credits go to their owners/developers. Their licenses are included in the corresponding folders.

Code in `beetl`: https://github.com/XiaoxiWei/NeurIPS_BEETL <br />
Code in `beetl`: https://github.com/sylvchev/beetl-competition <br />
Code in `moabb`: https://github.com/NeuroTechX/moabb <br />
Code in `braindecode`: https://github.com/braindecode/braindecode <br />
Code in `skorch`: https://github.com/skorch-dev/skorch <br />
Code in `braindecode/models/dann.py`: https://github.com/jvanvugt/pytorch-domain-adaptation <br />
