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

import h5py
import numpy as np

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.utils.dataset_helpers import make_group, store_dataset


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

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Write the neuron data to an HDF5 group.

        The method creates (or retrieves) a subgroup named after the neuron's
        ``index`` and stores the ``times`` array as a dataset called ``"times"``.
        The dataset is stored using 64‑bit floating point precision and gzip
        compression.

        Parameters
        ----------
        parent : h5py.Group
            The HDF5 group that will contain the neuron subgroup.

        Raises
        ------
        TypeError
            If ``self.times`` cannot be converted to a NumPy ``float64`` array.
        OSError
            If an error occurs while creating the group or writing the dataset
            (e.g., file is read‑only, insufficient disk space, or corrupted file).
        """
        # Group name is the neuron index (as a string)
        grp = make_group(parent, str(self.index))
        store_dataset(grp, "times", self.times.astype(np.float64), compression=True)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "Neuron":
        """Create a :class:`Neuron` instance from an HDF5 group.

        The function expects the group name to be the neuron's integer index and
        to contain a dataset named ``"times"`` holding the spike timestamps.

        Parameters
        ----------
        grp : h5py.Group
            The HDF5 group representing a single neuron.

        Returns
        -------
        Neuron
            A new instance populated with the index extracted from ``grp`` and
            the timestamps read from the ``"times"`` dataset.

        Raises
        ------
        ValueError
            If the group name cannot be parsed into an integer index.
        KeyError
            If the required ``"times"`` dataset is missing from ``grp``.
        OSError
            If an I/O error occurs while reading the dataset.
        """
        idx = int(grp.name.split("/")[-1])
        times = grp["times"][()]
        return cls(index=idx, times=times)