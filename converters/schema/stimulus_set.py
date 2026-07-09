"""Utility class for representing and persisting a collection of population‑level
stimuli in HDF5 files.  The primary public class, :class:`StimulusSet`, holds a
list of :class:`~converters.schema.population_stimulus.PopulationStimulus`
instances—one per cell type—and provides methods to serialize to and deserialize
from an :class:`h5py.Group`.

The module relies on the :class:`~converters.schema.hdf5_serializable.HDF5Serializable`
base class, which defines the required ``to_hdf5``/``from_hdf5`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import h5py

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.schema.population_stimulus import PopulationStimulus


@dataclass
class StimulusSet(HDF5Serializable):
    """Container for all stimulus blocks associated with a network simulation.

    Attributes:
        populations (List[PopulationStimulus]): A list containing one
            :class:`PopulationStimulus` for each cell‑type (e.g. dSPN, iSPN, …)
            that participates in the simulation.
    """

    populations: List[PopulationStimulus] = field(default_factory=list)

    def to_hdf5(self, stim_grp: h5py.Group) -> None:
        """Serialize the stimulus set into an HDF5 *stimulus* subgroup.

        The method creates (or re‑uses) the supplied ``stim_grp`` and delegates
        the actual serialization of each population to its own ``to_hdf5``
        implementation.

        Args:
            stim_grp: An open :class:`h5py.Group` that will receive the
                serialized data.  It is typically a subgroup named
                ``"stimulus"`` within a larger HDF5 file hierarchy.

        Raises:
            OSError: If the HDF5 file or group cannot be written to (e.g. due
                to insufficient permissions or a closed file handle).
            TypeError: If ``stim_grp`` is not an instance of :class:`h5py.Group`.
        """
        for pop in self.populations:
            pop.to_hdf5(stim_grp)

    @classmethod
    def from_hdf5(cls, stim_grp: h5py.Group) -> "StimulusSet":
        """Deserialize a stimulus set from an existing HDF5 *stimulus* group.

        The method iterates over all subgroup names found in ``stim_grp``,
        constructing a :class:`PopulationStimulus` for each using its
        ``from_hdf5`` class method.

        Args:
            stim_grp: An open :class:`h5py.Group` that contains the serialized
                stimulus data.  Each child of this group is expected to be a
                population‑specific subgroup interpretable by
                :class:`PopulationStimulus`.

        Returns:
            StimulusSet: An instance populated with the deserialized
                :class:`PopulationStimulus` objects.

        Raises:
            KeyError: If a required dataset or attribute is missing from a
                population subgroup.
            OSError: If the underlying HDF5 file cannot be read (e.g. corrupted
                file, I/O error).
            TypeError: If ``stim_grp`` is not an instance of :class:`h5py.Group`.
        """
        pops: List[PopulationStimulus] = []
        for pop_name in stim_grp.keys():
            pops.append(PopulationStimulus.from_hdf5(stim_grp, pop_name))
        return cls(populations=pops)