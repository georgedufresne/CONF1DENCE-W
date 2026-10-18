import h5py
import CONF1DENCE
import numpy as np
import json
import pandas as pd
import torch
import os
from torch.utils.data import Dataset

class readSEM(Dataset):
    """Read stellar evolution models stored in HDF5 files.

    The class stores only the path to the file and lazily opens the
    :class:`h5py.File` handle in ``__getitem__``.  This ensures that when used
    with :class:`torch.utils.data.DataLoader` each worker opens its own file
    handle and we avoid pickling an open file descriptor.
    """

    def __init__(self, **kwargs):
        # determine path to training files
        self.SEM_path = kwargs.get("SEM_path", None)
        if self.SEM_path is None:
            self.SEM_path = CONF1DENCE.__abspath__ + "data/HDF5_data/M15.h5"
        self.SEM_type = kwargs.get("SEM_type", "DSEP_iso")

        # read format file
        format_path = (
            CONF1DENCE.__abspath__ + "data/format/" + self.SEM_type + "_format.json"
        )
        with open(format_path, "r") as json_file:
            self.Data_format = json.load(json_file)

        # Attributes used for lazy loading
        self._h5file = None
        self._grp = None
        self.grp_name = None

        # select data loading method based on SEM_type and determine dataset length
        if self.SEM_type == "DSEP_iso":
            self.grp_name = kwargs.get("SBC", "PHOENIX")
            with h5py.File(self.SEM_path, "r") as f:
                self.n_samples = f[self.grp_name]["data"].shape[0]
            self.load_data = self.read_DSEP
        elif self.SEM_type == "DSEE_iso":
            with h5py.File(self.SEM_path, "r") as f:
                self.n_samples = f["data"].shape[0]
            self.load_data = self.read_DSEE
        else:
            raise Exception("SEM_type not supported")


    def _ensure_open(self):
        """Open the HDF5 file lazily when needed."""
        if self._h5file is None:
            self._h5file = h5py.File(self.SEM_path, "r", swmr=True)
            if self.SEM_type == "DSEP_iso":
                self._grp = self._h5file[self.grp_name]
            else:
                self._grp = self._h5file

    def close(self):
        """Close the HDF5 file if it is open."""
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None
            self._grp = None

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        self._ensure_open()
        data = self.load_data(idx)
        return data
        
    def read_DSEP(self, idx):
        """Read a single sample for ``DSEP_iso`` models."""
        return (
            torch.from_numpy(self._grp["data"][idx]),
            torch.from_numpy(self._grp["label"][idx]),
        )

    def read_DSEE(self, idx):
        """Read a single sample for ``DSEE_iso`` models."""
        return (
            torch.from_numpy(self._grp["data"][idx]),
            torch.from_numpy(self._grp["label"][idx]),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Do not pickle open file handles
        state["_h5file"] = None
        state["_grp"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # file handle will be opened lazily on first access

    def __del__(self):
        self.close()