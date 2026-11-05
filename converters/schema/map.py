# # -*- coding: utf-8 -*-
# #
# # SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# # SPDX-License-Identifier: Apache-2.0
# #

"""Utility classes for converting a collection of ``Population`` objects to and from an
HDF5 representation.

The module defines the :class:`Map` dataclass, which aggregates multiple
:class:`~converters.schema.population.Population` instances and provides methods to
serialize the whole collection to an HDF5 file and to reconstruct it from such a
file.

Typical usage::

    with h5py.File("output.h5", "w") as h5file:
        map_obj = Map(populations=[pop1, pop2])
        map_obj.to_hdf5(h5file)

    with h5py.File("output.h5", "r") as h5file:
        loaded_map = Map.from_hdf5(h5file)

Both methods rely on the :class:`~converters.schema.population.Population`
implementation of ``to_hdf5`` and ``from_hdf5``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import h5py

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.schema.population import Population
from converters.utils.dataset_helpers import make_group


@dataclass
class Map(HDF5Serializable):
    """Container for a collection of :class:`Population` objects that can be saved to
    and loaded from an HDF5 file.

    Attributes:
        populations: A list of :class:`Population` instances that belong to the map.
    """

    populations: List[Population] = field(default_factory=list)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Serialize the map and all of its populations into an HDF5 group.

        The method creates a subgroup named ``"Map"`` under ``parent`` and delegates
        the actual writing of each population to its own ``to_hdf5`` implementation.

        Args:
            parent: An open :class:`h5py.Group` (or file) that will contain the
                ``"Map"`` subgroup.

        Raises:
            OSError: If the HDF5 file cannot be written to (e.g., due to I/O errors).
            TypeError: If ``parent`` is not an ``h5py.Group`` instance.
            Exception: Propagates any exception raised by
                :meth:`Population.to_hdf5` or :func:`make_group`.
        """
        map_grp = make_group(parent, "Map")
        for pop in self.populations:
            pop.to_hdf5(map_grp)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "Map":
        """Reconstruct a :class:`Map` instance from an HDF5 group.

        The method expects a subgroup named ``"Map"`` inside ``grp`` and creates a
        :class:`Population` instance for each child group found therein.

        Args:
            grp: An open :class:`h5py.Group` (or file) that contains a ``"Map"``
                subgroup.

        Returns:
            A new :class:`Map` object populated with the deserialized
            :class:`Population` objects.

        Raises:
            KeyError: If the ``"Map"`` subgroup does not exist in ``grp``.
            OSError: If there is a problem reading from the HDF5 file.
            Exception: Propagates any exception raised by
                :meth:`Population.from_hdf5`.
        """
        pops = [Population.from_hdf5(pop_grp) for pop_grp in grp["Map"].values()]
        return cls(populations=pops)
