# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0

"""hdf5_interface.py
-------------------

Base interface for objects that can be written to and read from HDF5 files.

This module defines the :class:`HDF5Serializable` abstract base class which
specifies the minimal API required for an object to participate in a schema
that persists its state in an HDF5 hierarchy.  Implementations must provide
methods for serialising themselves under a given :class:`h5py.Group` and for
re‑creating an instance from an existing group.
"""

from abc import ABC, abstractmethod
from typing import Type, TypeVar

import h5py

# Generic type variable bound to HDF5Serializable for proper type hinting of
# the classmethod return type.
_T = TypeVar("_T", bound="HDF5Serializable")


def _decode(attr) -> str | None:
    """Return a clean text value for an HDF5 attribute.

    * If the attribute is ``bytes`` – decode as UTF‑8, replacing any illegal
      byte sequences.
    * If it is already a ``str`` – return it unchanged.
    * If the attribute is missing (``None``) – keep it as ``None``.
    """
    if isinstance(attr, bytes):
        return attr.decode("utf‑8", errors="replace")
    return attr  # str, None or any other python object


class HDF5Serializable(ABC):
    """Interface required for every object that participates in the schema.

    Sub‑classes must implement methods to serialise their state to an HDF5
    group and to restore an instance from such a group.  The methods are
    deliberately minimal to allow flexible implementation strategies while
    guaranteeing a consistent API.

    Example:
        ```python
        class MyData(HDF5Serializable):
            def __init__(self, value: int):
                self.value = value

            def to_hdf5(self, parent: h5py.Group) -> None:
                parent.create_dataset("value", data=self.value)

            @classmethod
            def from_hdf5(cls, grp: h5py.Group) -> "MyData":
                value = int(grp["value"][()])
                return cls(value)
        ```
    """

    @abstractmethod
    def to_hdf5(self, parent: h5py.Group) -> None:
        """Serialise the object **under** ``parent``.

        The implementation should create any necessary datasets, groups, or
        attributes within the supplied :class:`h5py.Group` to represent the
        object's state.  The caller is responsible for managing the lifecycle of
        the HDF5 file and the parent group.

        Args:
            parent: The HDF5 group that will contain the object's data.

        Raises:
            NotImplementedError: If the subclass does not provide an
                implementation (should never happen for concrete subclasses).
            h5py.H5Error: If an error occurs while writing to the HDF5 file
                (e.g., insufficient permissions or invalid data).
        """

    @classmethod
    @abstractmethod
    def from_hdf5(cls: Type[_T], grp: h5py.Group) -> _T:
        """Re‑create an instance from a pre‑existing HDF5 group.

        The method should read the data stored by :meth:`to_hdf5` and use it to
        construct a new instance of the class.

        Args:
            grp: The HDF5 group containing the serialized representation.

        Returns:
            An instance of the calling class reconstructed from ``grp``.

        Raises:
            NotImplementedError: If the subclass does not provide an
                implementation (should never happen for concrete subclasses).
            KeyError: If expected datasets, groups, or attributes are missing.
            h5py.H5Error: If an error occurs while reading from the HDF5 file.
        """
