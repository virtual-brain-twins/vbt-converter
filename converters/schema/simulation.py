# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0

"""Simulation schema utilities.

This module provides the :class:`Simulation` data class, which aggregates the
configuration, parameters, stimuli, and output map of a single simulation run.
Instances can be persisted to and restored from an HDF5 file using the
:class:`~converters.schema.hdf5_serializable.HDF5Serializable` interface.

Typical usage::

    sim = Simulation(simulation_id="001")
    sim.to_hdf5(h5file_root)

    # Later …
    loaded_sim = Simulation.from_hdf5(h5file_root["Sim_001"])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import h5py

from converters.schema.configuration import Configuration
from converters.schema.hdf5_serializable import HDF5Serializable
from converters.schema.map import Map
from converters.schema.simulation_parameters import SimulationParameters
from converters.schema.stimuli_group import StimuliGroup
from converters.utils.dataset_helpers import make_group


@dataclass
class Simulation(HDF5Serializable):
    """Container for all data describing a single simulation.

    Attributes
    ----------
    simulation_id: str
        Unique identifier for the simulation. It is used to name the top‑level
        HDF5 group (``Sim_<simulation_id>``).
    configuration: Configuration
        Simulation configuration information (e.g., hardware settings). A new
        :class:`Configuration` instance is created by default.
    parameters: SimulationParameters
        Numerical parameters governing the simulation. A new
        :class:`SimulationParameters` instance is created by default.
    stimuli: StimuliGroup
        Definition of the stimuli presented during the simulation. A new
        :class:`StimuliGroup` instance is created by default.
    output_map: Map
        Mapping of the simulation's output data. A new :class:`Map` instance is
        created by default.
    """

    simulation_id: str
    configuration: Configuration = field(default_factory=Configuration)
    parameters: SimulationParameters = field(default_factory=SimulationParameters)
    stimuli: StimuliGroup = field(default_factory=StimuliGroup)
    output_map: Map = field(default_factory=Map)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Serialize the simulation to an HDF5 group.

        The method creates a hierarchy under ``parent`` with the following
        structure::

            Sim_<simulation_id>/
                Configuration/
                Parameters/
                Stimuli/
                Output/

        Each sub‑object is responsible for writing its own contents.

        Parameters
        ----------
        parent: h5py.Group
            HDF5 group that will receive the ``Sim_<simulation_id>`` subgroup.

        Raises
        ------
        OSError
            If the HDF5 group cannot be created (e.g., due to I/O problems).
        Exception
            Propagates any exception raised by the ``to_hdf5`` methods of the
            constituent objects (``Configuration``, ``SimulationParameters``,
            ``StimuliGroup``, ``Map``).
        """
        sim_grp = make_group(parent, f"Sim_{self.simulation_id}")

        cfg_grp = make_group(sim_grp, "Configuration")
        self.configuration.to_hdf5(cfg_grp)

        param_grp = make_group(sim_grp, "Parameters")
        self.parameters.to_hdf5(param_grp)

        stim_grp = make_group(sim_grp, "Stimuli")
        self.stimuli.to_hdf5(stim_grp)

        out_grp = make_group(sim_grp, "Output")
        self.output_map.to_hdf5(out_grp)

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "Simulation":
        """Deserialize a simulation from an HDF5 group.

        The ``grp`` argument must correspond to a group named
        ``Sim_<simulation_id>`` and contain the sub‑groups ``Configuration``,
        ``Parameters``, ``Stimuli``, and ``Output`` as created by
        :meth:`to_hdf5`.

        Parameters
        ----------
        grp: h5py.Group
            HDF5 group representing a previously saved simulation.

        Returns
        -------
        Simulation
            An instance populated with data read from the HDF5 file.

        Raises
        ------
        KeyError
            If any of the required sub‑groups (``Configuration``, ``Parameters``,
            ``Stimuli``, ``Output``) are missing from ``grp``.
        OSError
            If there is a problem reading from the HDF5 file.
        ValueError
            If the group name does not contain a valid simulation identifier.
        Exception
            Propagates any exception raised by the ``from_hdf5`` methods of the
            constituent objects.
        """
        sim_id = grp.name.split("/")[-1].replace("Sim_", "")
        if not sim_id:
            raise ValueError("Simulation group name does not contain an identifier.")
        cfg = Configuration.from_hdf5(grp["Configuration"])
        params = SimulationParameters.from_hdf5(grp["Parameters"])
        stimuli = StimuliGroup.from_hdf5(grp["Stimuli"])
        out_map = Map.from_hdf5(grp["Output"])

        return cls(
            simulation_id=sim_id,
            configuration=cfg,
            parameters=params,
            stimuli=stimuli,
            output_map=out_map,
        )
