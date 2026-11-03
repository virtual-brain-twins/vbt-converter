# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0
#
"""Utility module for handling the *StandardFormat* HDF5 schema.

The :class:`StandardFormatGroup` class represents the root of the new schema
(``/StandardFormat``) and provides methods to serialize the in‑memory representation
to an :class:`h5py.File` and to reconstruct it back from such a file.

Typical usage
--------------

>>> import h5py
>>> from converters.schema.standard_format_group import StandardFormatGroup
>>> with h5py.File("example.h5", "w") as f:
...     group = StandardFormatGroup()
...     group.to_hdf5(f)
...
>>> with h5py.File("example.h5", "r") as f:
...     loaded = StandardFormatGroup.from_hdf5(f)
"""

from dataclasses import dataclass, field
from typing import List

import h5py

from converters.schema.global_section import GlobalSection
from converters.schema.hdf5_serializable import HDF5Serializable
from converters.schema.simulation import Simulation
from converters.utils.dataset_helpers import make_group


@dataclass
class StandardFormatGroup(HDF5Serializable):
    """Root container for the *StandardFormat* HDF5 schema.

    This dataclass stores a single :class:`~converters.schema.global_section.GlobalSection`
    instance and an ordered list of :class:`~converters.schema.simulation.Simulation`
    objects.  It can write itself to an HDF5 file and be reconstructed from one.

    Attributes
    ----------
    global_section : GlobalSection
        The global configuration section of the file.  It is created lazily
        with a default ``GlobalSection`` instance if none is supplied.
    runs : List[Simulation]
        A list of simulation runs to be stored under the ``/StandardFormat/Simulations``
        group.

    Methods
    -------
    to_hdf5(root: h5py.File) -> None
        Serialises the instance into the provided HDF5 file.
    from_hdf5(root: h5py.File) -> StandardFormatGroup
        Creates a new instance from the contents of an HDF5 file.
    """

    global_section: GlobalSection = field(default_factory=GlobalSection)
    runs: List[Simulation] = field(default_factory=list)

    def to_hdf5(self, root: h5py.File) -> None:
        """Serialise this :class:`StandardFormatGroup` into an HDF5 file.

        The method creates (or re‑uses) the ``/StandardFormat`` group, writes the
        contained :class:`~converters.schema.global_section.GlobalSection` and then
        creates a ``Simulations`` subgroup in which each :class:`~converters.schema.simulation.Simulation`
        is stored.

        Args
        ----
        root:
            An open :class:`h5py.File` instance opened in a mode that permits
            writing (e.g. ``'w'`` or ``'a'``).

        Raises
        ------
        OSError
            If the HDF5 file cannot be written to, e.g. due to I/O errors or
            insufficient permissions.
        RuntimeError
            If any of the contained objects fail to serialise themselves.
        """
        std_grp = make_group(root, "StandardFormat")
        self.global_section.to_hdf5(std_grp)

        sims_grp = make_group(std_grp, "Simulations")
        for sim in self.runs:
            sim.to_hdf5(sims_grp)

    @classmethod
    def from_hdf5(cls, root: h5py.File) -> "StandardFormatGroup":
        """Create a :class:`StandardFormatGroup` from an HDF5 file.

        The method expects a ``StandardFormat`` group at the root of the file,
        with a ``Global`` subgroup for the global section and a ``Simulations``
        subgroup containing one subgroup per simulation run.

        Args
        ----
        root:
            An open :class:`h5py.File` instance opened in a mode that permits
            reading (e.g. ``'r'``).

        Returns
        -------
        StandardFormatGroup
            A populated instance reconstructed from the file contents.

        Raises
        ------
        KeyError
            If any of the required groups (``StandardFormat``, ``Global`` or
            ``Simulations``) are missing from the file.
        OSError
            If the file cannot be read, for example because it is corrupted or
            closed.
        RuntimeError
            If any of the sub‑objects fail to deserialize correctly.
        """
        std_grp = root["StandardFormat"]
        glob = GlobalSection.from_hdf5(std_grp["Global"])
        sims = [Simulation.from_hdf5(g) for g in std_grp["Simulations"].values()]
        return cls(global_section=glob, runs=sims)
