# NeurIPS 2021 BEETL Competition - Benchmarks for EEG Transfer Learning
** The this the package for loading competition data **

## Competition information

All details about the competition are available on the official website: https://beetl.ai/

We offer two specific challenges to promote the development of EEG decoding to use big data:

* Task 1 is a cross-subject sleep stage decoding challenge reflecting the need for transfer learning in clinical diagnostics. This challenge aims to compare and give a benchmark of transfer learning algorithms in the literature.
* Task 2 is a cross-dataset motor imagery decoding challenge reflecting the need for transfer learning in human interfacing. This challenge aims to promote new algorithms that can utilise EEG data across data sets or data centers.

## Getting started

You could use your own [Conda](https://www.anaconda.com/products/individual) environment or create a new one with:

```
$ conda create --name beetl numpy scikit-learn=0.23
$ conda activate beetl
$ pip install git+https://github.com/sylvchev/beetl-competition
```

If you use your own conda environment, you just need to type `pip install git+https://github.com/sylvchev/beetl-competition` or `pip install -e git+https://github.com/sylvchev/beetl-competition#egg=beetl-competition`

## How to start

Once beetl is installed, you could download the data with, 'path' returns the folder name contains the data

```
from beetl.task_datasets import BeetlSleepTutorial, BeetlSleepSource, BeetlSleepLeaderboard, BeetlMILeaderboard
ds = BeetlSleepTutorial()
path = ds.download()
X, y, info = ds.get_data()
```
