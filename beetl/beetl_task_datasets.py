import os
import glob
import os.path as osp
import shutil

import numpy as np
from mne import set_config
from moabb.datasets.download import (
    fs_get_file_hash,
    fs_get_file_id,
    fs_get_file_list,
    get_dataset_path,
)
from moabb.utils import set_download_dir
from pooch import HTTPDownloader, Unzip, retrieve

BEETL_URL = "https://ndownloader.figshare.com/files/"

import mne
mne.set_log_level('CRITICAL')

class BeetlDataset:
    def __init__(self, figshare_id, code, subject_list):
        self.figshare_id = figshare_id
        self.code = code
        self.subject_list = subject_list

    def data_path(self, subject):
        pass

    def download(self, path=None, subjects=None):
        """Download datasets for sleep task

        Parameters
        ----------
        path: str | None
            Path to download the data, store in ~/mne_data if None
        subjects: list | None
            list of subject, default=None to select all subjects

        Returns
        --------
        path: str
            path to the downloaded data
        """
        if path:
            set_download_dir(path)
            set_config("MNE_DATASETS_{}_PATH".format(self.code.upper()), path)
            # TODO: Fix FileExistsError: [Errno 17] File exists: '/Users/X/test_compet in
            # moabb/utils.py in set_download_dir(path), l. 54

        subjects = self.subject_list if subjects is None else subjects
        # Competition files
        spath = []
        for s in subjects:
            spath.append(self.data_path(s))
        return osp.dirname(spath[-1][0])

    def get_data(self, subjects=None):
        pass

class BeetlMILeaderboard(BeetlDataset):
    def __init__(self):
        super().__init__(
            figshare_id=14839650,
            code="beetlMIleaderboard",
            subject_list=range(1, 6),
        )

    def data_path(self, subject):
        sign = self.code
        key_dest = "MNE-{:s}-data".format(sign.lower())
        path = osp.join(get_dataset_path(sign, None), key_dest)

        filelist = fs_get_file_list(self.figshare_id)
        reg = fs_get_file_hash(filelist)
        fsn = fs_get_file_id(filelist)
        for f in fsn.keys():
            if not osp.exists(osp.join(path, "S1", "training", "race1_padsData.npy")):
                retrieve(
                    BEETL_URL + fsn[f],
                    reg[fsn[f]],
                    fsn[f],
                    path,
                    processor=Unzip(),
                    downloader=HTTPDownloader(progressbar=True),
                )
                zpath = osp.join(path, fsn[f] + ".unzip", "leaderboardMI")
                for s in range(1, 6):
                    os.mkdir(osp.join(path, "S{}".format(s)))
                    os.mkdir(osp.join(path, "S{}".format(s), "training"))
                    os.mkdir(osp.join(path, "S{}".format(s), "testing"))
                for s in self.subject_list:
                    zptr = osp.join(zpath, "S{}".format(s), "training")
                    zpte = osp.join(zpath, "S{}".format(s), "testing")
                    ptr = osp.join(path, "S{}".format(s), "training")
                    pte = osp.join(path, "S{}".format(s), "testing")
                    if s < 3:
                        for i in range(1, 6):
                            fx = "race{}_padsData.npy".format(i)
                            fy = "race{}_padsLabel.npy".format(i)
                            shutil.move(osp.join(zptr, fx), osp.join(ptr, fx))
                            shutil.move(osp.join(zptr, fy), osp.join(ptr, fy))
                        for i in range(6, 16):
                            fx = "race{}_padsData.npy".format(i)
                            shutil.move(osp.join(zpte, fx), osp.join(pte, fx))
                    else:
                        fx = "training_s{}X.npy".format(s)
                        fy = "training_s{}y.npy".format(s)
                        tfx = "testing_s{}X.npy".format(s)
                        shutil.move(osp.join(zptr, fx), osp.join(ptr, fx))
                        shutil.move(osp.join(zptr, fy), osp.join(ptr, fy))
                        shutil.move(osp.join(zpte, tfx), osp.join(pte, tfx))
                    os.rmdir(zptr)
                    os.rmdir(zpte)
                    zpths = osp.join(zpath, "S{}".format(s))
                    os.rmdir(zpths)
                os.rmdir(osp.join(path, fsn[f] + ".unzip", "leaderboardMI"))
                os.rmdir(osp.join(path, fsn[f] + ".unzip"))
        spath = []
        ptr = osp.join(path, "S{}".format(subject), "training")
        pte = osp.join(path, "S{}".format(subject), "testing")
        if subject < 3:
            for i in range(1, 6):
                fx = "race{}_padsData.npy".format(i)
                fy = "race{}_padsLabel.npy".format(i)
                spath.append(osp.join(ptr, fx))
                spath.append(osp.join(ptr, fy))
            for i in range(6, 16):
                fx = "race{}_padsData.npy".format(i)
                spath.append(osp.join(pte, fx))
        else:
            fx = "training_s{}X.npy".format(subject)
            fy = "training_s{}y.npy".format(subject)
            tfx = "testing_s{}X.npy".format(subject)
            spath.append(osp.join(ptr, fx))
            spath.append(osp.join(ptr, fy))
            spath.append(osp.join(pte, tfx))
        return spath

    def get_data(self, path=None, subjects=None, dataset=None):
        """Get data as list of numpy array, labels and metadata

        Parameters
        ----------
        path: str | None
            Path to download the data, store in ~/mne_data if None
        dataset: str
            'A' or 'B' for leaderboard datasets

        Returns
        --------
        X_target: ndarray, shape (n_trials, n_electrodes, n_samples)
            ndarray for labeled EEG signal
        y_target: ndarray, shape (n_trials)
            label for the EEG signal
        X_testing: ndarray, shape (n_trials, n_electrodes, n_samples)
            ndarray for unlabeled EEG signal
        """

        spath = []
        for s in subjects:
            files = self.data_path(s)
            for f in files:
                spath.append(f)
        X_target, y_target, X_testing = [], [], []
        for p in spath:
            d = np.load(p, allow_pickle=True)
            if osp.basename(p)[-5] == "l" or osp.basename(p)[-5] == "y":
                y_target.append(d)
            elif osp.basename(p).startswith("training"):
                X_target.append(d)
            elif osp.basename(p).startswith("testing"):
                X_testing.append(d)
            elif int(osp.basename(p)[4]) < 6 and osp.basename(p)[5] == "_":
                X_target.append(d)
            else:
                X_testing.append(d)
        X_target = np.concatenate(X_target)
        X_testing = np.concatenate(X_testing)
        y_target = np.concatenate(y_target)
        return X_target, y_target, X_testing

class Cybathlon(BeetlDataset):
    def __init__(self):
        super().__init__(
            figshare_id=00000000, # TODO : update this to be correct
            code="beetlmitest",
            subject_list=range(1, 6),
        )

    def data_path(self, subject):
        sign = self.code
        key_dest = "MNE-{:s}-data".format(sign.lower())
        path = osp.join(get_dataset_path(sign, None), key_dest)

        spath = []
        ptr = osp.join(path, "S{}".format(subject), "training")
        pte = osp.join(path, "S{}".format(subject), "testing")
        if subject < 4:
            for i in range(1, 6):
                fx = "race{}_padsData.npy".format(i)
                fy = "race{}_padsLabel.npy".format(i)
                spath.append(osp.join(ptr, fx))
                spath.append(osp.join(ptr, fy))
            for i in range(6, 16):
                fx = "race{}_padsData.npy".format(i)
                spath.append(osp.join(pte, fx))
        else:
            fx = "training_s{}X.npy".format(subject)
            fy = "training_s{}y.npy".format(subject)
            tfx = "testing_s{}X.npy".format(subject)
            spath.append(osp.join(ptr, fx))
            spath.append(osp.join(ptr, fy))
            spath.append(osp.join(pte, tfx))
        return spath

    def get_data(self, path=None, subjects=None, dataset=None):
        """Get data as list of numpy array, labels and metadata

        Parameters
        ----------
        path: str | None
            Path to download the data, store in ~/mne_data if None
        dataset: str
            'A' or 'B' for leaderboard datasets

        Returns
        --------
        X_target: ndarray, shape (n_trials, n_electrodes, n_samples)
            ndarray for labeled EEG signal
        y_target: ndarray, shape (n_trials)
            label for the EEG signal
        X_testing: ndarray, shape (n_trials, n_electrodes, n_samples)
            ndarray for unlabeled EEG signal
        """

        spath = []
        for s in subjects:
            files = self.data_path(s)
            for f in files:
                spath.append(f)
        X_target, y_target, X_testing = [], [], []
        for p in spath:
            d = np.load(p, allow_pickle=True)
            if osp.basename(p)[-5] == "l" or osp.basename(p)[-5] == "y":
                y_target.append(d)
            elif osp.basename(p).startswith("training"):
                X_target.append(d)
            elif osp.basename(p).startswith("testing"):
                X_testing.append(d)
            elif int(osp.basename(p)[4]) < 6 and osp.basename(p)[5] == "_":
                X_target.append(d)
            else:
                X_testing.append(d)
        X_target = np.concatenate(X_target)
        X_testing = np.concatenate(X_testing)
        y_target = np.concatenate(y_target)

        return X_target, y_target, X_testing

class BeetlSleepTutorial(BeetlDataset):
    def __init__(self):
        super().__init__(
            figshare_id=14779407,
            code="beetlsleeptutorial",
            subject_list=range(10),
        )

    def data_path(self, subject):
        sign = self.code
        key_dest = "MNE-{:s}-data".format(sign.lower())
        path = osp.join(get_dataset_path(sign, None), key_dest)

        filelist = fs_get_file_list(self.figshare_id)
        reg = fs_get_file_hash(filelist)
        fsn = fs_get_file_id(filelist)
        spath = []
        for f in fsn.keys():
            if not osp.exists(osp.join(path, "s{}r1X.npy".format(subject))):
                retrieve(
                    BEETL_URL + fsn[f],
                    reg[fsn[f]],
                    fsn[f],
                    path,
                    processor=Unzip(),
                    downloader=HTTPDownloader(progressbar=True),
                )
                zpath = osp.join(path, fsn[f] + ".unzip")
                for i in self.subject_list:
                    fx, fy = "s{}r1X.npy".format(i), "s{}r1y.npy".format(i)
                    shutil.move(osp.join(zpath, fx), osp.join(path, fx))
                    shutil.move(osp.join(zpath, fy), osp.join(path, fy))
                shutil.move(
                    osp.join(zpath, "headerInfo.npy"), osp.join(path, "headerInfo.npy")
                )
                os.rmdir(osp.join(path, fsn[f] + ".unzip"))
        spath.append(osp.join(path, "s{}r1X.npy".format(subject)))
        spath.append(osp.join(path, "s{}r1y.npy".format(subject)))
        spath.append(osp.join(path, "headerInfo.npy"))
        return spath

    def get_data(self, path=None, subjects=None):
        """Get data as list of numpy array, labels and metadata

        Parameters
        ----------
        path: str | None
            Path to download the data, store in ~/mne_data if None
        subjects: list | None
            list of subject, default=None to select all subjects

        Returns
        --------
        X: ndarray, shape (n_trials, n_electrodes, n_samples)
            ndarray for EEG signal
        y: ndarray, shape (n_trials)
            label for the EEG signal
        metadata: mne.Info
            metadata of acquisition as mne.Info
        """
        subjects = self.subject_list if subjects is None else subjects
        spath = []
        for s in subjects:
            files = self.data_path(s)
            for f in files:
                if osp.basename(f) != "headerInfo.npy":
                    spath.append(f)
                else:
                    hd = f
        spath.append(hd)
        X, y, meta = [], [], []
        for p in spath:
            d = np.load(p, allow_pickle=True)
            if osp.basename(p)[4] == "X":
                X.append(d)
            elif osp.basename(p)[4] == "y":
                y.append(d)
            elif osp.basename(p) == "headerInfo.npy":
                meta = d
        X = np.concatenate(X)
        y = np.concatenate(y)
        return X, y, meta


class BeetlSleepSource(BeetlDataset):
    def __init__(self):
        super().__init__(
            figshare_id=14839659,
            code="beetlsleepsource",
            subject_list=range(39),
        )

    def data_path(self, subject):
        sign = self.code
        key_dest = "MNE-{:s}-data".format(sign.lower())
        path = osp.join(get_dataset_path(sign, None), key_dest)

        filelist = fs_get_file_list(self.figshare_id)
        reg = fs_get_file_hash(filelist)
        fsn = fs_get_file_id(filelist)
        spath = []
        for f in fsn.keys():
            if not osp.exists(osp.join(path, "training_s{}r1X.npy".format(subject))):
                retrieve(
                    BEETL_URL + fsn[f],
                    reg[fsn[f]],
                    fsn[f],
                    path,
                    processor=Unzip(),
                    downloader=HTTPDownloader(progressbar=True),
                )
                zpath = osp.join(path, fsn[f] + ".unzip", "SleepSource")
                for i in self.subject_list:
                    for s in [1, 2]:
                        fx, fy = (
                            "training_s{}r{}X.npy".format(i, s),
                            "training_s{}r{}y.npy".format(i, s),
                        )
                        shutil.move(osp.join(zpath, fx), osp.join(path, fx))
                        shutil.move(osp.join(zpath, fy), osp.join(path, fy))
                shutil.move(
                    osp.join(zpath, "headerInfo.npy"), osp.join(path, "headerInfo.npy")
                )
                os.rmdir(osp.join(path, fsn[f] + ".unzip", "SleepSource"))
                os.rmdir(osp.join(path, fsn[f] + ".unzip"))
        for s in [1, 2]:
            spath.append(osp.join(path, "training_s{}r{}X.npy".format(subject, s)))
            spath.append(osp.join(path, "training_s{}r{}y.npy".format(subject, s)))
        spath.append(osp.join(path, "headerInfo.npy"))
        return spath

    def get_data(self, path=None, subjects=None):
        """Get data as list of numpy array, labels and metadata

        Parameters
        ----------
        path: str | None
            Path to download the data, store in ~/mne_data if None
        subjects: list | None
            list of subject, default=None to select all subjects

        Returns
        --------
        X: ndarray, shape (n_trials, n_electrodes, n_samples)
            ndarray for EEG signal
        y: ndarray, shape (n_trials)
            label for the EEG signal
        metadata: mne.Info
            metadata of acquisition as mne.Info
        """
        subjects = self.subject_list if subjects is None else subjects
        spath = []
        for s in subjects:
            files = self.data_path(s)
            for f in files:
                if osp.basename(f) != "headerInfo.npy":
                    spath.append(f)
                else:
                    hd = f
        spath.append(hd)
        X, y, meta = [], [], []
        for p in spath:
            d = np.load(p, allow_pickle=True)
            if osp.basename(p).split(".")[0][-1] == "X":
                X.append(d)
            elif osp.basename(p).split(".")[0][-1] == "y":
                y.append(d)
            elif osp.basename(p) == "headerInfo.npy":
                meta = d
        X = np.concatenate(X)
        y = np.concatenate(y)
        return X, y, meta


class BeetlSleepLeaderboard(BeetlDataset):
    def __init__(self):
        super().__init__(
            figshare_id=14839653,
            code="beetlsleepleaderboard",
            subject_list=range(18),
        )

    def data_path(self, subject):
        sign = self.code
        key_dest = "MNE-{:s}-data".format(sign.lower())
        path = osp.join(get_dataset_path(sign, None), key_dest)

        filelist = fs_get_file_list(self.figshare_id)
        reg = fs_get_file_hash(filelist)
        fsn = fs_get_file_id(filelist)
        spath = []
        for f in fsn.keys():
            if not osp.exists(osp.join(path, "sleep_target", "leaderboard_s0r1X.npy")):
                retrieve(
                    BEETL_URL + fsn[f],
                    reg[fsn[f]],
                    fsn[f],
                    path,
                    processor=Unzip(),
                    downloader=HTTPDownloader(progressbar=True),
                )
                zpath = osp.join(path, fsn[f] + ".unzip", "LeaderboardSleep")
                os.mkdir(osp.join(path, "sleep_target"))
                os.mkdir(osp.join(path, "testing"))
                zptr = osp.join(zpath, "sleep_target")
                ptr = osp.join(path, "sleep_target")
                zpte = osp.join(zpath, "testing")
                pte = osp.join(path, "testing")
                for i in self.subject_list:
                    for s in [1, 2]:
                        if i < 6:
                            fx = "leaderboard_s{}r{}X.npy".format(i, s)
                            fy = "leaderboard_s{}r{}y.npy".format(i, s)
                            shutil.move(osp.join(zptr, fx), osp.join(ptr, fx))
                            shutil.move(osp.join(zptr, fy), osp.join(ptr, fy))
                        else:
                            fx = "leaderboard_s{}r{}X.npy".format(i, s)
                            shutil.move(osp.join(zpte, fx), osp.join(pte, fx))
                hi = "headerInfo.npy"
                shutil.move(osp.join(zptr, hi), osp.join(ptr, hi))
                os.rmdir(zptr)
                os.rmdir(zpte)
                os.rmdir(zpath)
                os.rmdir(osp.join(path, fsn[f] + ".unzip"))
        for s in [1, 2]:
            if subject < 6:
                fd = "sleep_target"
                fy = "leaderboard_s{}r{}y.npy".format(subject, s)
                spath.append(osp.join(path, fd, fy))
            else:
                fd = "testing"
            fx = "leaderboard_s{}r{}X.npy".format(subject, s)
            spath.append(osp.join(path, fd, fx))
        spath.append(osp.join(path, "sleep_target", "headerInfo.npy"))
        return spath

    def get_data(self, path=None, subjects=None):
        """Get data as list of numpy array, labels and metadata

        Parameters
        ----------
        path: str | None
            Path to download the data, store in ~/mne_data if None
        subjects: list | None
            list of subject, default=None to select all subjects

        Returns
        --------
        X_target: ndarray, shape (n_trials, n_electrodes, n_samples)
            ndarray for labeled EEG signal
        y_target: ndarray, shape (n_trials)
            label for the EEG signal
        X_testing: ndarray, shape (n_trials, n_electrodes, n_samples)
            ndarray for unlabeled EEG signal
        metadata: mne.Info
            metadata of acquisition as mne.Info
        """
        subjects = self.subject_list if subjects is None else subjects
        spath = []
        for s in subjects:
            files = self.data_path(s)
            for f in files:
                if osp.basename(f) != "headerInfo.npy":
                    spath.append(f)
                else:
                    hd = f
        spath.append(hd)
        X_target, y_target, X_testing, meta = [], [], [], []
        for p in spath:
            d = np.load(p, allow_pickle=True)
            if osp.basename(p).split(".")[0][-1] == "X":
                if int(osp.basename(p).split("s")[1].split("r")[0]) < 6:
                    X_target.append(d)
                else:
                    X_testing.append(d)
            elif osp.basename(p).split(".")[0][-1] == "y":
                y_target.append(d)
            elif osp.basename(p) == "headerInfo.npy":
                meta = d
        X_target = np.concatenate(X_target) if len(X_target) > 0 else np.array(X_target)
        X_testing = (
            np.concatenate(X_testing) if len(X_testing) > 0 else np.array(X_testing)
        )
        y_target = np.concatenate(y_target) if len(y_target) > 0 else np.array(y_target)
        return X_target, y_target, X_testing, meta