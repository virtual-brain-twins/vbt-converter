"""Utilities for storing external dependency information in HDF5 files.

This module defines the :class:`ExternalDependenciesGrp` dataclass, which provides
a convenient way to serialize a simple ``name → version / url`` mapping into an
HDF5 file. The mapping is stored as attributes of the ``/ExternalDependencies``
group, but the implementation can easily be extended to use a small JSON dataset
if desired.

Typical usage
-------------
>>> grp = ExternalDependenciesGrp(deps={"numpy": "1.26.0", "pandas": "2.2.0"})
>>> grp.to_hdf5(h5file)          # Serialize to an HDF5 file
>>> restored = ExternalDependenciesGrp.from_hdf5(h5file)  # Deserialize
>>> assert restored.deps == grp.deps
"""

from dataclasses import dataclass, field
from typing import Dict

import h5py

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.utils.dataset_helpers import make_group


@dataclass
class ExternalDependenciesGrp(HDF5Serializable):
    """Container for external dependencies stored under ``/ExternalDependencies``.

    The group stores a dictionary mapping dependency names to a version string or
    URL. The contents are written as HDF5 attributes, which makes them easy to
    inspect with standard HDF5 tools.

    Attributes:
        deps: Mapping of dependency identifiers (e.g., package name) to a
            version string, URL, or any other textual descriptor.
    """

    deps: Dict[str, str] = field(default_factory=dict)

    def to_hdf5(self, root: h5py.File) -> None:
        """Write the dependency mapping to an HDF5 file.

        A group named ``ExternalDependencies`` is created (or retrieved) under the
        supplied ``root`` file, and each key/value pair in :attr:`deps` is stored
        as an attribute of that group.

        Args:
            root: An open :class:`h5py.File` object that will receive the new
                ``ExternalDependencies`` group.

        Raises:
            OSError: If the file cannot be written to (e.g., due to I/O errors).
            ValueError: If the provided ``root`` is not a valid HDF5 file object.
            RuntimeError: Propagated from :func:`make_group` when the group cannot
                be created.
        """
        dep_grp = make_group(root, "ExternalDependencies")
        for k, v in self.deps.items():
            dep_grp.attrs[k] = v

    @classmethod
    def from_hdf5(cls, root: h5py.File) -> "ExternalDependenciesGrp":
        """Create an instance from an existing HDF5 file.

        The method reads the attributes of the ``ExternalDependencies`` group and
        populates the :attr:`deps` dictionary.

        Args:
            root: An open :class:`h5py.File` object containing an
                ``ExternalDependencies`` group.

        Returns:
            ExternalDependenciesGrp: A new instance populated with the data read
                from the file.

        Raises:
            KeyError: If the ``ExternalDependencies`` group does not exist in the
                supplied file.
            OSError: If the file cannot be read (e.g., due to I/O errors).
            ValueError: If attribute values cannot be decoded as strings.
        """
        grp = root["ExternalDependencies"]
        return cls(deps={k: str(v) for k, v in grp.attrs.items()})