# # -*- coding: utf-8 -*-
# #
# # SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# # SPDX-License-Identifier: Apache-2.0
# #

"""
This module defines :class:`GlobalSection`, a concrete implementation of the
:class:`~converters.schema.hdf5_serializable.HDF5Serializable` protocol that
stores global-level metadata for a conversion job.  The class can serialize its
contents to an HDF5 group and reconstruct itself from such a group using the
helper functions defined in :mod:`converters.utils.compound_attribute` and
:mod:`converters.utils.dataset_helpers`.

Typical usage
-------------
>>> section = GlobalSection(
...     metadata_mapping={"temperature": (300, Unit("K"))},
...     parameter_simulation_mapping={"p1": "sim_001"},
...     experiment_info="Test run",
... )
>>> section.to_hdf5(root_group)          # Serialize to HDF5
>>> loaded = GlobalSection.from_hdf5(root_group)   # Deserialize
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional

import h5py

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.utils.common import read_compound_attr_parsed
from converters.utils.compound_attribute_helper import write_compound_attr
from converters.utils.dataset_helpers import make_group
from converters.utils.units import Unit


@dataclass
class GlobalSection(HDF5Serializable):
    """
    Container for global metadata, experiment information and parameter‑simulation
    mappings that can be written to / read from an HDF5 file.

    Attributes
    ----------
    metadata_mapping : Dict[Any, Tuple[Any, Optional[Unit]]]
        Arbitrary key‑value pairs supplied by the user.  Each value may be
        accompanied by an optional :class:`~converters.utils.units.Unit`.  The
        mapping is stored as a set of compound attributes in the ``Global``
        group.

    parameter_simulation_mapping : Dict[str, str]
        A dictionary mapping a textual description of a parameter combination
        (e.g. ``"p1"``, ``"p2"``, …) to a simulation identifier.  This information is
        written as a compound attribute without an associated unit.

    experiment_info : str
        Free‑form text describing the experiment.  Stored as a compound attribute
        without a unit.
    """

    metadata_mapping: Dict[Any, Tuple[Any, Optional[Unit]]] = field(default_factory=dict)
    parameter_simulation_mapping: Dict[str, str] = field(default_factory=dict)
    experiment_info: str = ""

    def to_hdf5(self, parent: h5py.Group) -> None:
        """
        Serialize the instance into a sub‑group named ``"Global"`` under the
        supplied *parent* HDF5 group.

        The method creates the ``Global`` group (or re‑uses an existing one) and
        writes three categories of data:

        * **User‑supplied key‑value pairs** – each entry of
          :attr:`metadata_mapping` is written as a compound attribute using
          :func:`~converters.utils.compound_attribute.write_compound_attr`.
        * **Experiment information** – stored under the attribute name
          ``"experiment_info"`` with no unit.
        * **Parameter‑simulation mapping** – stored under the attribute name
          ``"parameter_simulation_mapping"``, also without a unit.

        Parameters
        ----------
        parent : h5py.Group
            The HDF5 group that will contain the ``Global`` subgroup.
        """
        grp = make_group(parent, "Global")

        # user supplied kv‑pairs (value + unit)
        for key, (val, unit) in self.metadata_mapping.items():
            write_compound_attr(grp, key, val, unit)

        # mandatory special keys
        # experiment_info has no unit
        write_compound_attr(grp, "experiment_info", self.experiment_info, None)

        # parameter_simulation_mapping is a dictionary mapping parameter combinations to simulation IDs,
        # does not have any unit
        write_compound_attr(
            grp, "parameter_simulation_mapping", self.parameter_simulation_mapping, None
        )

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "GlobalSection":
        """
        Re‑construct a :class:`GlobalSection` instance from a previously
        serialized ``Global`` HDF5 group.

        The method reads all compound attributes in *grp* except the two
        reserved names ``"experiment_info"`` and
        ``"parameter_simulation_mapping"``.  Those reserved attributes are read
        separately because they are required fields but do not carry units.

        Parameters
        ----------
        grp : h5py.Group
            The HDF5 group that contains the serialized ``Global`` subsection.

        Returns
        -------
        GlobalSection
            An instance populated with the metadata mapping, experiment info,
            and parameter‑simulation mapping extracted from *grp*.
        """
        mapping: Dict[Any, Tuple[Any, Optional[Unit]]] = {}
        for attr_name in grp.attrs:
            if attr_name in ("experiment_info", "parameter_simulation_mapping"):
                continue
            val, unit = read_compound_attr_parsed(grp, attr_name)
            mapping[attr_name] = (val, unit)

        exp_info, _ = read_compound_attr_parsed(grp, "experiment_info")
        param_map, _ = read_compound_attr_parsed(grp, "parameter_simulation_mapping")
        return cls(
            metadata_mapping=mapping,
            parameter_simulation_mapping=param_map,
            experiment_info=exp_info,
        )
