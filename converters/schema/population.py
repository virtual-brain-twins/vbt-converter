# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0

"""Utilities for representing a population of neurons and serializing them to/from HDF5.

This module defines the :class:`Population` dataclass, which groups a collection
of :class:`~converters.schema.neuron.Neuron` instances.  It provides methods to
store the population in an HDF5 file hierarchy and to reconstruct a population
from such a hierarchy.
"""

from dataclasses import dataclass, field
from typing import List

import h5py

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.schema.neuron import Neuron
from converters.utils.dataset_helpers import make_group


@dataclass
class Population(HDF5Serializable):
    """One population – stores a list of ``Neuron`` objects.

    Attributes:
        population_name: The name of the population; used as the HDF5 group name.
        neurons: A list of :class:`~converters.schema.neuron.Neuron` objects that
            belong to this population.
    """

    population_name: str
    neurons: List[Neuron] = field(default_factory=list)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Serialize the population into an HDF5 group.

        A new subgroup with the name ``population_name`` is created under
        ``parent``.  Each :class:`Neuron` in ``neurons`` is then asked to
        serialize itself into that subgroup.

        Args:
            parent: An open :class:`h5py.Group` that will receive the new
                population subgroup.

        Raises:
            TypeError: If ``parent`` is not an instance of :class:`h5py.Group`.
            OSError: If an I/O error occurs while creating the group or writing
                data (e.g., insufficient permissions or corrupted file).
        """
        pop_grp = make_group(parent, self.population_name)
        for neuron in self.neurons:
            neuron.to_hdf5(pop_grp)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "Population":
        """Reconstruct a :class:`Population` instance from an HDF5 group.

        The function expects each subgroup of ``grp`` to contain a serialized
        :class:`Neuron`.  It loads each neuron and assembles them into a new
        :class:`Population` object, using the final component of ``grp``'s name
        as the population name.

        Args:
            grp: An HDF5 group whose sub‑groups each represent a serialized
                :class:`Neuron`.

        Returns:
            A populated :class:`Population` instance.

        Raises:
            TypeError: If ``grp`` is not an instance of :class:`h5py.Group`.
            OSError: If an I/O error occurs while reading the group.
            KeyError: If expected neuron sub‑groups are missing or malformed.
            ValueError: If the extracted population name is invalid.
        """
        neurons = [Neuron.from_hdf5(neu_grp) for neu_grp in grp.values()]
        return cls(population_name=grp.name.split("/")[-1], neurons=neurons)
