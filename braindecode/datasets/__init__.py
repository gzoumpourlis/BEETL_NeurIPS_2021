"""
Loader code for some datasets.
"""
from .base import WindowsDataset, BaseDataset, BaseConcatDataset
from .moabb import MOABBDataset
from .custom_dataset import CustomDataset