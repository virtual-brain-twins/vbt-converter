"""Utilities for storing simulation‑wide parameters in an HDF5 file.

This module defines a simple key/value container that can be serialized to
and deserialized from an ``h5py.Group``.  All values are stored as HDF5
attributes; scalars are written directly while more complex objects are JSON‑
encoded before being saved.  The container is implemented as a dataclass that
inherits from :class:`~converters.schema.hdf5_serializable.HDF5Serializable`.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any

import h5py
import numpy as np

from converters.schema.hdf5_serializable import HDF5Serializable


@dataclass
class SimulationParameters(HDF5Serializable):
    """A free‑form collection of simulation parameters.

    The parameters are stored as a dictionary of arbitrary JSON‑serialisable
    Python objects.  When written to an HDF5 file each key becomes an attribute
    of the target group.

    Attributes:
        kv_pairs: Mapping of parameter names to values.  The values may be any
            JSON‑serialisable type or a NumPy scalar.
    """

    kv_pairs: Dict[str, Any] = field(default_factory=dict)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Write the stored key/value pairs as attributes of an HDF5 group.

        Simple scalar types (``int``, ``float``, ``str``, ``bytes`` and
        ``numpy.number``) are stored directly.  All other objects are JSON‑
        encoded and stored as strings.

        Args:
            parent: The HDF5 group whose attributes will receive the data.

        Raises:
            TypeError: If a non‑scalar value cannot be JSON‑encoded by
                :func:`json.dumps`.
        """
        for k, v in self.kv_pairs.items():
            # HDF5 can store simple scalars directly; everything else is JSON‑encoded.
            if isinstance(v, (int, float, str, bytes, np.number)):
                parent.attrs[k] = v
            else:
                parent.attrs[k] = json.dumps(v)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "SimulationParameters":
        """Create a :class:`SimulationParameters` instance from an HDF5 group.

        Attributes stored as JSON strings are decoded back into their original
        Python objects.  If an attribute cannot be decoded, its raw value is
        retained.

        Args:
            grp: The HDF5 group containing the stored attributes.

        Returns:
            A new :class:`SimulationParameters` instance populated with the
            retrieved key/value pairs.

        Raises:
            OSError: If reading attributes from ``grp`` fails (e.g., the group
                does not exist or is inaccessible).
        """
        out: Dict[str, Any] = {}
        for k, v in grp.attrs.items():
            try:
                out[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                out[k] = v
        return cls(kv_pairs=out)
