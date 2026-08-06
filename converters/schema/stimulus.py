"""
Data model for a piece‑wise constant stimulus and utilities for serialising it
to and from HDF5 files.

The :class:`Stimulus` class stores the limits and values that define a
piece‑wise constant waveform, optional physical units for each array, and a
string identifying the stimulus type.  It inherits from
:class:`~converters.schema.hdf5_serializable.HDF5Serializable` which defines the
``to_hdf5`` and ``from_hdf5`` interface used throughout the converters package.

Typical usage::

    stim = Stimulus(
        limits=np.array([0.0, 1.0, 2.0]),
        values=np.array([5.0, 10.0]),
        stim_type="piecewise_constant",
        limits_unit=Unit("s"),
        values_unit=Unit("mV")
    )
    with h5py.File("stim.h5", "w") as f:
        grp = f.create_group("Stimulus")
        stim.to_hdf5(grp)

    # ...later...
    with h5py.File("stim.h5", "r") as f:
        loaded = Stimulus.from_hdf5(f["Stimulus"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import h5py
import numpy as np

from converters.schema.hdf5_serializable import HDF5Serializable, _decode
from converters.utils.dataset_helpers import store_dataset
from converters.utils.units import Unit


@dataclass
class Stimulus(HDF5Serializable):
    """Container for a piece‑wise constant stimulus.

    Attributes
    ----------
    limits : np.ndarray
        One‑dimensional array of the stimulus segment boundaries.  It must have
        a length exactly one greater than ``values``.
    values : np.ndarray
        One‑dimensional array of the stimulus amplitude for each segment.
    stim_type : str
        Identifier for the type of stimulus (e.g. ``"piecewise_constant"``).
    limits_unit : Optional[Unit]
        Physical unit of the ``limits`` array, stored as an HDF5 attribute if
        provided.
    values_unit : Optional[Unit]
        Physical unit of the ``values`` array, stored as an HDF5 attribute if
        provided.
    """

    limits: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    values: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    stim_type: str = "piecewise_constant"
    limits_unit: Optional[Unit] = None
    values_unit: Optional[Unit] = None

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Serialise the stimulus to an HDF5 group.

        The method creates three datasets within ``parent``:

        * ``"Limits"`` – the ``limits`` array (no compression)
        * ``"Values"`` – the ``values`` array (no compression)
        * ``"StimType"`` – a scalar, variable‑length UTF‑8 string dataset

        If ``limits_unit`` and/or ``values_unit`` are supplied, they are written
        as the ``"unit"`` attribute on the corresponding datasets.

        Parameters
        ----------
        parent : h5py.Group
            The HDF5 group that will receive the new datasets.
        """
        store_dataset(parent, "Limits", self.limits, compression=False)
        if self.limits_unit is not None:
            parent["Limits"].attrs["unit"] = self.limits_unit

        store_dataset(parent, "Values", self.values, compression=False)
        if self.values_unit is not None:
            parent["Values"].attrs["unit"] = self.values_unit

        # StimType is a variable‑length UTF‑8 string dataset
        dt = h5py.string_dtype()
        ds = parent.create_dataset("StimType", shape=(), dtype=dt)
        ds[()] = self.stim_type

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "Stimulus":
        """Create a ``Stimulus`` instance from an HDF5 group.

        The method reads the ``Limits`` and ``Values`` datasets, together with
        the optional ``unit`` attributes, and the scalar ``StimType`` dataset.
        Basic validation ensures the dimensionality and length relationship
        required by the piece‑wise constant representation.

        Parameters
        ----------
        grp : h5py.Group
            Group containing the ``Limits``, ``Values`` and ``StimType`` datasets.

        Returns
        -------
        Stimulus
            An instance populated with the data extracted from ``grp``.

        Raises
        ------
        ValueError
            If ``limits`` or ``values`` are not one‑dimensional, or if their
            lengths do not satisfy ``len(limits) == len(values) + 1``.
        """
        limits = grp["Limits"][()]
        values = grp["Values"][()]
        stim_type = _decode(grp["StimType"][()])

        # optional unit attributes
        lim_u = _decode(grp["Limits"].attrs.get("unit"))
        val_u = _decode(grp["Values"].attrs.get("unit"))

        # Basic validation (unchanged)
        if limits.ndim != 1 or values.ndim != 1:
            raise ValueError("Both 'limits' and 'values' must be one‑dimensional arrays.")
        if limits.size != values.size + 1:
            raise ValueError(
                f"'limits' length ({limits.size}) must be exactly one more than "
                f"'values' length ({values.size})."
            )

        return cls(
            limits=limits,
            values=values,
            stim_type=stim_type,
            limits_unit=lim_u,
            values_unit=val_u,
        )
