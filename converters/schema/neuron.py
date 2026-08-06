"""Utility definitions for storing neuronal spike times in HDF5 files.

This module provides the :class:`Neuron` dataclass, which encapsulates a
single neuron's identifier and its spike timestamps.  The class implements the
:class:`~converters.schema.hdf5_serializable.HDF5Serializable` protocol, allowing
instances to be written to and restored from an HDF5 group using the helper
functions :func:`~converters.utils.dataset_helpers.make_group` and
:func:`~converters.utils.dataset_helpers.store_dataset`.

Typical usage
-------------
>>> neuron = Neuron(index=5, times=np.array([0.1, 0.3, 0.7]))
>>> with h5py.File("spikes.h5", "w") as f:
...     neuron.to_hdf5(f)

>>> with h5py.File("spikes.h5", "r") as f:
...     loaded = Neuron.from_hdf5(f["5"])
"""

from dataclasses import dataclass, field
from typing import Optional

import h5py
import numpy as np

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.utils.dataset_helpers import make_group, store_dataset
from converters.utils.units import Unit


@dataclass
class Neuron(HDF5Serializable):
    """Container for a single neuron's spike times.

    Attributes
    ----------
    index : int
        Unique identifier for the neuron.  Used as the name of the HDF5 group
        when the object is serialized.
    times : np.ndarray
        One‑dimensional array of spike timestamps (float64).  Empty by default.
    """

    index: int
    times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    times_unit: Optional[Unit] = None

    def to_hdf5(self, parent: h5py.Group) -> None:
        grp = make_group(parent, str(self.index))
        store_dataset(grp, "times", self.times.astype(np.float64), compression=True)
        if self.times_unit is not None:
            grp["times"].attrs["unit"] = self.times_unit

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "Neuron":
        idx = int(grp.name.split("/")[-1])
        times = grp["times"][()]
        unit = grp["times"].attrs.get("unit")
        return cls(index=idx, times=times, times_unit=unit)
