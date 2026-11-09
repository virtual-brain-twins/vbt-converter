# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for serialising and deserialising collections of :class:`~converters.schema.stimulus.Stimulus`
objects together with per‑stimulus metadata and optional population‑list information to/from
HDF5 files.

The module defines :class:`StimuliGroup`, a thin wrapper around three dictionaries that map
sub‑group names to (i) a :class:`Stimulus` object, (ii) an attribute map containing scalar
metadata, and (iii) an optional list of population identifiers.  The group can be written to an
HDF5 ``Group`` via :meth:`StimuliGroup.to_hdf5` and reconstructed from an HDF5 ``Group`` via
:func:`StimuliGroup.from_hdf5`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import h5py

from converters.schema.stimulus import Stimulus
from converters.utils.common import read_compound_attr_parsed
from converters.utils.compound_attribute_helper import write_compound_attr
from converters.utils.units import Unit


@dataclass
class StimuliGroup:
    """Container for a set of stimuli and their associated metadata.

    Attributes
    ----------
    stimuli_map : Dict[str, Stimulus]
        Mapping from a sub‑group name to the corresponding :class:`Stimulus` instance.
    attr_map : Dict[str, Dict[str, Tuple[Any, Optional[Unit]]]]
        Mapping from a sub‑group name to a dictionary of scalar metadata.  Each inner
        dictionary maps an attribute key to a ``(value, unit)`` tuple where *unit* may be
        ``None``.  Values are written as *compound attributes* using the helper functions
        :func:`write_compound_attr` and :func:`read_compound_attr`.
    pop_list_map : Dict[str, List[str]]
        Optional mapping from a sub‑group name to a list of population identifiers.  If a
        list is present it is stored as a dataset called ``"PopulationList"`` in the
        respective HDF5 sub‑group.
    """

    stimuli_map: Dict[str, Stimulus] = field(default_factory=dict)
    attr_map: Dict[str, Dict[str, Tuple[Any, Optional[Unit]]]] = field(default_factory=dict)
    pop_list_map: Dict[str, List[str]] = field(default_factory=dict)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Serialise the group into an HDF5 ``Group``.

        Each entry in :attr:`stimuli_map` creates a new sub‑group under *parent*.
        The corresponding :class:`Stimulus` object is asked to serialise itself into that
        sub‑group, after which any scalar metadata from :attr:`attr_map` is written as
        compound attributes.  If a population list exists for the sub‑group it is stored
        as a dataset named ``"PopulationList"``.

        Parameters
        ----------
        parent : h5py.Group
            The HDF5 group that will receive the sub‑groups representing each stimulus.
        """
        for sub_name, stim in self.stimuli_map.items():
            sub_grp = parent.create_group(sub_name)
            stim.to_hdf5(sub_grp)

            # write scalar meta‑data as compound attributes
            for key, (val, unit) in self.attr_map.get(sub_name, {}).items():
                # ``val`` may be a plain scalar or a JSON‑serializable object.
                write_compound_attr(sub_grp, key, val, unit)
            pop_list = self.pop_list_map.get(sub_name, None)
            if pop_list is not None:
                sub_grp.create_dataset("PopulationList", data=pop_list,
                                       dtype=h5py.string_dtype())

    @classmethod
    def from_hdf5(cls, parent: h5py.Group) -> "StimuliGroup":
        """Reconstruct a :class:`StimuliGroup` from an HDF5 ``Group``.

        The method iterates over each sub‑group in *parent*, reads the :class:`Stimulus`
        instance via ``Stimulus.from_hdf5`` and extracts any compound attributes using
        :func:`read_compound_attr`.  The unit component of the attribute tuple is ignored
        because the metadata is treated as unit‑agnostic during deserialisation.  If a
        ``"PopulationList"`` dataset is present its contents are loaded as a list of
        strings.

        Parameters
        ----------
        parent : h5py.Group
            The HDF5 group that contains the stimulus sub‑groups.

        Returns
        -------
        StimuliGroup
            An instance populated with the stimuli, attribute maps, and optional
            population lists extracted from *parent*.
        """
        stimuli_map: Dict[str, Stimulus] = {}
        attr_map: Dict[str, Dict[str, Tuple[Any, Optional[Unit]]]] = {}
        pop_list_map: Dict[str, List[str]] = {}

        for sub_name in parent.keys():
            sub_grp = parent[sub_name]
            stimuli_map[sub_name] = Stimulus.from_hdf5(sub_grp)

            # read back the *compound* attributes
            attrs: Dict[str, Any] = {}
            for attr_key in sub_grp.attrs:
                val, _ = read_compound_attr_parsed(sub_grp, attr_key)  # unit is ignored for meta‑data
                attrs[attr_key] = val
            attr_map[sub_name] = attrs

            # Read optional population list
            if "PopulationList" in sub_grp:
                pop_list_map[sub_name] = sub_grp["PopulationList"][:].astype(str).tolist()

        return cls(stimuli_map=stimuli_map, attr_map=attr_map, pop_list_map=pop_list_map)
