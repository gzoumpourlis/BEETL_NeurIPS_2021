"""
A dataset handle and abstract low level access to the data. the dataset will
takes data stored locally, in the format in which they have been downloaded,
and will convert them into a MNE raw object. There are options to pool all the
different recording sessions per subject or to evaluate them separately.
"""
# flake8: noqa
from .bnci import (
    BNCI2014001,
    BNCI2014002,
    BNCI2014004,
    BNCI2014008,
    BNCI2014009,
    BNCI2015001,
    BNCI2015003,
    BNCI2015004,
)
from .gigadb import Cho2017
from .physionet_mi import PhysionetMI
