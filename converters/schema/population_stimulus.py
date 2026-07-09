"""
Utilities for serialising :class:`~converters.schema.stimulus.Stimulus` objects
to and from HDF5 files.

The module provides helper functions to write arbitrary Python objects as HDF5
attributes (scalar values are stored directly, while non‑scalar values are JSON
encoded).  It also defines a :class:`PopulationStimulus` dataclass that groups
multiple stimulus sub‑sources belonging to a neuronal population and handles
their HDF5 representation.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict

import h5py
import numpy as np

from converters.schema.stimulus import Stimulus


def _write_attribute(parent: h5py.Group, name: str, value: Any) -> None:
    """Write a value as an HDF5 attribute of *parent*.

    Scalars (including NumPy scalar types) are written directly.  Non‑scalar
    objects such as dictionaries, lists, tuples or custom objects are JSON‑
    encoded and stored as a variable‑length UTF‑8 string.

    Args:
        parent: The HDF5 group whose attribute dictionary will be modified.
        name: The name of the attribute to create or overwrite.
        value: The Python object to store.

    Raises:
        TypeError: If *value* cannot be JSON‑encoded.
        OSError: If the attribute cannot be created in the underlying HDF5 file.
    """
    if isinstance(value, (str, bytes, np.generic, int, float, bool)):
        # scalar / native type → write directly
        parent.attrs[name] = value
    else:
        # Anything else (dict, list, tuple, custom object) → JSON string
        json_str = json.dumps(value, ensure_ascii=False)
        # h5py needs an explicit variable‑length string dtype
        vlen_str_dtype = h5py.string_dtype()
        parent.attrs.create(name, json_str, dtype=vlen_str_dtype)


def _read_attribute(parent: h5py.Group, name: str) -> Any:
    """Read an attribute from *parent*, decoding JSON when appropriate.

    The function first retrieves the raw attribute value.  It then attempts to
    decode the value as JSON; if decoding fails (because the attribute was a
    plain scalar), the original value is returned unchanged.

    Args:
        parent: The HDF5 group containing the attribute.
        name: The name of the attribute to read.

    Returns:
        The decoded Python object if the attribute contained JSON, otherwise the
        raw scalar value.

    Raises:
        KeyError: If *name* is not present in ``parent.attrs``.
        OSError: If there is a problem accessing the HDF5 file.
    """
    raw = parent.attrs[name]
    # Try to decode JSON – if it fails we assume it was a plain scalar
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw


@dataclass
class PopulationStimulus:
    """Container for a neuronal population’s stimulus sub‑sources.

    Each population (e.g. ``dSPN``) may consist of several stimulus sub‑sources
    such as cortical input or GPe‑FoxP2.  Every sub‑source is represented by a
    :class:`~converters.schema.stimulus.Stimulus` instance plus an optional
    dictionary of scalar meta‑data.  The meta‑data are stored as HDF5 attributes
    of the corresponding sub‑group.

    Attributes:
        name: Human‑readable identifier of the population (e.g. ``"dSPN"``).
        sub_stimuli: Mapping from sub‑source name to the corresponding
            :class:`Stimulus` object.
        sub_attrs: Mapping from sub‑source name to a dictionary of scalar
            meta‑data that will be written as HDF5 attributes.
    """

    name: str  # e.g. "dSPN"
    sub_stimuli: Dict[str, Stimulus] = field(default_factory=dict)
    sub_attrs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # sub_attrs['cortical'] = {"generator": "poisson", "cluster_size": 5, …}

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Serialise the population into an HDF5 group.

        A subgroup named after ``self.name`` is created under *parent*.  Inside
        this group, each sub‑source gets its own subgroup that stores the
        ``Stimulus`` data.  Any scalar meta‑data supplied in ``sub_attrs`` are
        written as attributes of the corresponding sub‑group.

        Args:
            parent: The HDF5 group that will receive the population subgroup.

        Raises:
            OSError: If creation of groups or attributes fails.
            Exception: Propagates any exception raised by ``Stimulus.to_hdf5``.
        """
        pop_grp = parent.create_group(self.name)

        for sub_name, stim in self.sub_stimuli.items():
            sub_grp = pop_grp.create_group(sub_name)
            stim.to_hdf5(sub_grp)

            # Write any scalar meta‑data as attributes of the sub‑group.
            for key, val in self.sub_attrs.get(sub_name, {}).items():
                # HDF5 attributes can only store scalars, so we JSON‑encode any
                # non‑scalar values.
                _write_attribute(sub_grp, key, val)

    @classmethod
    def from_hdf5(cls, parent: h5py.Group, name: str) -> "PopulationStimulus":
        """Reconstruct a :class:`PopulationStimulus` instance from HDF5.

        The function expects an existing group ``parent[name]`` that follows the
        layout produced by :meth:`to_hdf5`.  It reads the stimulus sub‑sources
        and extracts all attributes of each sub‑group into ``sub_attrs``.

        Args:
            parent: The HDF5 group containing the population subgroup.
            name: The name of the population subgroup to read.

        Returns:
            A fully populated :class:`PopulationStimulus` object.

        Raises:
            KeyError: If ``name`` does not exist in *parent* or expected
                attributes are missing.
            OSError: If there is an issue accessing the underlying HDF5 file.
            Exception: Propagates any exception raised by
                ``Stimulus.from_hdf5``.
        """
        pop_grp = parent[name]
        sub_stimuli: Dict[str, Stimulus] = {}
        sub_attrs: Dict[str, Dict[str, Any]] = {}

        for sub_name in pop_grp.keys():
            sub_grp = pop_grp[sub_name]
            sub_stimuli[sub_name] = Stimulus.from_hdf5(sub_grp)
            # Copy all attributes of the sub‑group into a dict.
            sub_attrs[sub_name] = {
                k: _read_attribute(sub_grp, k) for k in sub_grp.attrs.keys()
            }

        return cls(name=name, sub_stimuli=sub_stimuli, sub_attrs=sub_attrs)
