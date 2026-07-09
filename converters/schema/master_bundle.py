"""Master bundle handling for HDF5 conversion.

This module defines :class:`MasterBundle`, a high‑level container that aggregates
the three main components produced by ``convert_any_to_hdf5``:

* :class:`~converters.schema.standard_format_group.StandardFormatGroup` – the
  scientific data in the standard format;
* :class:`~converters.schema.raw_bundle_group.RawBundleGroup` – the original
  input data packaged as a zip archive (optional);
* :class:`~converters.schema.external_dependencies_group.ExternalDependenciesGrp`
  – optional version‑tracking information for external dependencies.

The bundle can be written to or read from an HDF5 file that follows schema
version ``1.0``.
"""

from dataclasses import dataclass, field
from pathlib import Path

import h5py

from converters.schema.external_dependencies_group import ExternalDependenciesGrp
from converters.schema.hdf5_serializable import HDF5Serializable
from converters.schema.raw_bundle_group import RawBundleGroup
from converters.schema.standard_format_group import StandardFormatGroup


@dataclass
class MasterBundle(HDF5Serializable):
    """Container that bundles standard data, raw input, and optional dependencies.

    Attributes
    ----------
    standard : StandardFormatGroup
        The scientific data formatted according to the standard schema.
    raw : RawBundleGroup | None
        Optional raw input data packaged as a zip archive.
    deps : ExternalDependenciesGrp | None
        Optional information about external dependencies and their versions.
    """

    standard: StandardFormatGroup = field(default_factory=StandardFormatGroup)
    raw: RawBundleGroup | None = None
    deps: ExternalDependenciesGrp | None = None

    def to_hdf5(self, out_path: Path) -> None:
        """Write the bundle to an HDF5 file.

        The method creates the parent directory of ``out_path`` if it does not
        already exist, opens the file in write mode, stores a ``format_version``
        attribute, and delegates the actual serialization to the contained
        groups.

        Parameters
        ----------
        out_path : pathlib.Path
            Destination path for the generated HDF5 file.

        Raises
        ------
        OSError
            If the directory cannot be created or the file cannot be written.
        h5py.HDF5Error
            If an error occurs while interacting with the HDF5 library.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as hf:
            hf.attrs["format_version"] = "1.0"  # bump – now a new schema version
            self.standard.to_hdf5(hf)
            if self.raw:
                self.raw.to_hdf5(hf)
            if self.deps:
                self.deps.to_hdf5(hf)

    @classmethod
    def from_hdf5(cls, in_path: Path) -> "MasterBundle":
        """Create a :class:`MasterBundle` instance from an HDF5 file.

        The method reads the standard format group and, if present, the raw
        bundle and external‑dependency groups, then returns a fully populated
        ``MasterBundle`` object.

        Parameters
        ----------
        in_path : pathlib.Path
            Path to an existing HDF5 file that follows the expected schema.

        Returns
        -------
        MasterBundle
            An instance populated with the data read from ``in_path``.

        Raises
        ------
        FileNotFoundError
            If ``in_path`` does not exist.
        OSError
            If the file cannot be opened for reading.
        h5py.HDF5Error
            If an error occurs while reading the HDF5 contents.
        """
        with h5py.File(in_path, "r") as hf:
            std = StandardFormatGroup.from_hdf5(hf)
            raw = RawBundleGroup.from_hdf5(hf) if "RawBundle" in hf else None
            deps = ExternalDependenciesGrp.from_hdf5(hf) if "ExternalDependencies" in hf else None
            return cls(standard=std, raw=raw, deps=deps)