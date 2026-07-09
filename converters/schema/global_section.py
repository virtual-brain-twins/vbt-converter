"""Global section handling for the StandardFormat HDF5 schema.

This module provides the :class:`GlobalSection` dataclass, an implementation of
:class:`~converters.schema.hdf5_serializable.HDF5Serializable` that stores
metadata as HDF5 attributes under the ``/StandardFormat/Global/`` group.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

import h5py

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.utils.dataset_helpers import make_group


@dataclass
class GlobalSection(HDF5Serializable):
    """Container for global metadata stored as HDF5 attributes.

    The ``/StandardFormat/Global/`` group holds only attributes (the legacy
    *metadata*).  ``metadata_mapping`` is a free‑form dictionary that will be
    written to the group as string attributes, while ``experiment_info`` is a
    dedicated attribute.

    Attributes:
        metadata_mapping: Mapping of arbitrary keys to string values that will be
            saved as attributes on the HDF5 group. Keys are converted to strings
            before serialization.
        experiment_info: A free‑form string describing the experiment; stored
            under the ``experiment_info`` attribute.
    """

    metadata_mapping: Dict[Any, str] = field(default_factory=dict)
    experiment_info: str = ""

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Write the section's contents to an HDF5 group.

        A ``Global`` subgroup is created (or retrieved) under ``parent`` and the
        contents of :attr:`metadata_mapping` and :attr:`experiment_info` are
        stored as string attributes.

        Args:
            parent: The HDF5 group that will contain the ``Global`` subgroup.

        Raises:
            OSError: If the HDF5 file cannot be written (e.g., I/O errors).
            TypeError: If a value cannot be converted to a string for storage.
        """
        grp = make_group(parent, "Global")
        for key, val in self.metadata_mapping.items():
            # Keys are converted to strings – HDF5 attributes are always strings.
            grp.attrs[str(key)] = str(val)
        grp.attrs["experiment_info"] = self.experiment_info

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> "GlobalSection":
        """Create a :class:`GlobalSection` instance from an existing HDF5 group.

        The method reads all attributes of ``grp`` (except ``experiment_info``)
        into ``metadata_mapping`` and extracts ``experiment_info`` separately.

        Args:
            grp: The HDF5 group representing ``/StandardFormat/Global/`` from
                which to load data.

        Returns:
            An instance of :class:`GlobalSection` populated with the attributes
            read from ``grp``.

        Raises:
            OSError: If reading from the HDF5 file fails.
            KeyError: If expected attributes are missing (unlikely, as missing
                attributes are treated as empty strings).
        """
        mapping = {k: str(v) for k, v in grp.attrs.items() if k != "experiment_info"}
        info = grp.attrs.get("experiment_info", "")
        return cls(metadata_mapping=mapping, experiment_info=str(info))