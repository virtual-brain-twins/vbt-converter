"""This module defines the :class:`Stimulus` data class – a lightweight container for a
piecewise‑constant stimulus (or any other type of stimulus that can be described by
arrays of limits and corresponding values).  The class implements the
:class:`~converters.schema.hdf5_serializable.HDF5Serializable` protocol, allowing a
stimulus instance to be written to or reconstructed from an HDF5 group using
`h5py`.

Typical usage
~~~~~~~~~~~~~
>>> stim = Stimulus(
...     limits=np.array([0.0, 1.0, 2.0]),
...     values=np.array([0.5, 1.5]),
...     max_value=1.5,
...     stim_type="piecewise_constant",
... )
>>> with h5py.File("stim.h5", "w") as f:
...     grp = f.create_group("stimulus")
...     stim.to_hdf5(grp)
>>> with h5py.File("stim.h5", "r") as f:
...     loaded = Stimulus.from_hdf5(f["stimulus"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import h5py
import numpy as np

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.utils.dataset_helpers import store_dataset


@dataclass
class Stimulus(HDF5Serializable):
    """Container for a stimulus that can be serialized to HDF5.

    Attributes
    ----------
    limits : np.ndarray
        A one‑dimensional array defining the boundaries (edges) of the stimulus
        intervals.  Shape ``(N + 1,)`` where ``N`` is the number of intervals.
    values : np.ndarray
        A one‑dimensional array of values associated with each interval defined by
        ``limits``.  Shape ``(N,)``.
    max_value : float
        The maximum value observed in ``values``; stored separately for quick access.
    stim_type : str
        A string identifier for the type of stimulus (e.g. ``"piecewise_constant"``).

    Notes
    -----
    The class inherits from :class:`~converters.schema.hdf5_serializable.HDF5Serializable`,
    requiring implementation of :meth:`to_hdf5` and :meth:`from_hdf5`.
    """

    limits: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    values: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    max_value: float = 0.0
    stim_type: str = "piecewise_constant"

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Write the stimulus data to an HDF5 group.

        Each attribute is stored as a separate dataset inside ``parent``.  The
        datasets are written without compression to keep the operation fast.

        Args
        ----
        parent : h5py.Group
            The HDF5 group that will receive the stimulus datasets.  The group
            must be writable.

        Raises
        ------
        OSError
            If the HDF5 file underlying ``parent`` cannot be written to (e.g.
            due to I/O errors or insufficient permissions).
        ValueError
            If any of the attributes cannot be converted to a NumPy array suitable
            for storage.
        """
        store_dataset(parent, "Limits", self.limits, compression=False)
        store_dataset(parent, "Values", self.values, compression=False)
        store_dataset(parent, "MaxValue", np.array(self.max_value), compression=False)

    @classmethod
    def from_hdf5(cls: Type["Stimulus"], grp: h5py.Group) -> "Stimulus":
        """Reconstruct a :class:`Stimulus` instance from an HDF5 group.

        The method expects three datasets – ``Limits``, ``Values`` and
        ``MaxValue`` – to be present in ``grp``.  Missing datasets or type
        mismatches will raise an exception.

        Args
        ----
        grp : h5py.Group
            The HDF5 group containing the stimulus datasets.

        Returns
        -------
        Stimulus
            A new instance populated with the data read from ``grp``.

        Raises
        ------
        KeyError
            If any of the required datasets (``Limits``, ``Values``, ``MaxValue``)
            are missing from ``grp``.
        TypeError
            If the data cannot be cast to the expected NumPy types.
        ValueError
            If the arrays have incompatible shapes (e.g. ``limits`` does not have
            ``len(values) + 1`` elements).
        """
        limits = grp["Limits"][()]
        values = grp["Values"][()]
        max_val = float(grp["MaxValue"][()])

        # Basic validation to provide clearer error messages.
        if limits.ndim != 1 or values.ndim != 1:
            raise ValueError("Both 'limits' and 'values' must be one‑dimensional arrays.")
        if limits.size != values.size + 1:
            raise ValueError(
                f"'limits' length ({limits.size}) must be exactly one more than "
                f"'values' length ({values.size})."
            )

        return cls(limits=limits, values=values, max_value=max_val)
