"""SpikeMap module.

This module defines the :class:`SpikeMap` data class, which serves as the top‑level
container for collections of :class:`~converters.schema.population.Population`
objects. It provides methods to serialize the container and its contents to an
HDF5 file and to reconstruct the container from an HDF5 group.

Typical usage::

    from converters.schema.spike_map import SpikeMap
    import h5py

    # Create a SpikeMap instance
    spike_map = SpikeMap(populations=[pop1, pop2])

    # Write to an HDF5 file
    with h5py.File("data.h5", "w") as f:
        spike_map.to_hdf5(f)

    # Read from an HDF5 file
    with h5py.File("data.h5", "r") as f:
        loaded_map = SpikeMap.from_hdf5(f)

"""
from dataclasses import dataclass, field
from typing import List

import h5py

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.schema.population import Population
from converters.utils.dataset_helpers import make_group


@dataclass
class SpikeMap(HDF5Serializable):
    """Top‑level container that groups all populations.

    Attributes:
        populations: A list of :class:`~converters.schema.population.Population`
            instances contained in the spike map.
    """

    populations: List[Population] = field(default_factory=list)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Serialize the spike map to an HDF5 group.

        The method creates a ``SpikeMap`` subgroup under ``parent`` (or retrieves an
        existing one) and then delegates the serialization of each contained
        population to its own ``to_hdf5`` method.

        Args:
            parent: An open :class:`h5py.Group` that will receive the ``SpikeMap``
                subgroup.

        Raises:
            OSError: If the HDF5 file cannot be written to (e.g., due to I/O
                errors or insufficient permissions).
            KeyError: If a required subgroup cannot be created because a
                conflicting name already exists.
        """
        map_grp = make_group(parent, "SpikeMap")
        for pop in self.populations:
            pop.to_hdf5(map_grp)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "SpikeMap":
        """Reconstruct a :class:`SpikeMap` instance from an HDF5 group.

        The method expects the provided ``grp`` to contain a ``SpikeMap`` subgroup
        whose child groups represent individual populations. Each population group
        is deserialized via :meth:`Population.from_hdf5`.

        Args:
            grp: An open :class:`h5py.Group` that contains a ``SpikeMap`` subgroup.

        Returns:
            A new :class:`SpikeMap` instance populated with the deserialized
            :class:`Population` objects.

        Raises:
            KeyError: If the ``SpikeMap`` subgroup or any expected population subgroup
                is missing from ``grp``.
            TypeError: If the data within a population subgroup cannot be interpreted
                as a valid :class:`Population`.
            OSError: If there is an error reading from the HDF5 file.
        """
        pops = [Population.from_hdf5(pop_grp) for pop_grp in grp["SpikeMap"].values()]
        return cls(populations=pops)
