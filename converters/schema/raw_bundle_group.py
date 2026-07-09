"""converters.schema.raw_bundle_group

This module defines :class:`RawBundleGroup`, a lightweight wrapper that stores a
single external HDF5 link to a ``raw_bundle.zip`` file.  The class implements the
:class:`~converters.schema.hdf5_serializable.HDF5Serializable` protocol, allowing
the object to be written to and restored from an HDF5 file.

Typical usage::

    bundle = RawBundleGroup(out_path=Path("/tmp"))
    bundle.to_hdf5(h5_root)

    # Later …
    restored = RawBundleGroup.from_hdf5(h5_root)

Only the external link is persisted; the actual zip file remains outside of the
HDF5 container.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import h5py

from converters.schema.external_link import ExternalLink
from converters.schema.hdf5_serializable import HDF5Serializable
from converters.utils.dataset_helpers import make_group


@dataclass
class RawBundleGroup(HDF5Serializable):
    """Represent the ``/RawBundle`` group containing a single external zip link.

    The group contains one scalar dataset named ``raw_bundle.zip`` that stores
    an :class:`~converters.schema.external_link.ExternalLink` pointing to a zip
    archive (relative to the HDF5 file location).  The default link targets
    ``./raw_bundle.zip``.

    Attributes:
        zip_link: An :class:`ExternalLink` instance pointing at the zip file.
        out_path: The directory where the external zip file resides.  The
            :meth:`to_hdf5` method changes the current working directory to this
            path before attaching the link.
    """

    # The external link that will be written into the HDF5 file.
    zip_link: ExternalLink = field(
        default_factory=lambda: ExternalLink("./raw_bundle.zip")
    )
    # Directory containing the external zip file.
    out_path: Path = field(default_factory=Path)

    def to_hdf5(self, root: h5py.File) -> None:
        """Write the bundle information into an HDF5 file.

        The method creates (or retrieves) a group named ``RawBundle`` under
        ``root`` and attaches the external link as a scalar dataset called
        ``raw_bundle.zip``.  The current working directory is temporarily changed
        to :attr:`out_path` so that the relative link resolves correctly.

        Args:
            root: An open :class:`h5py.File` object representing the HDF5 file to
                which the information should be written.

        Raises:
            OSError: If changing the working directory to ``out_path`` fails.
            RuntimeError: If the external link cannot be attached (e.g., the
                dataset already exists with an incompatible type).
            h5py.HDF5Error: Propagated from the underlying HDF5 library when group
                creation or dataset operations fail.
        """
        # Change to the directory containing the external zip file.
        os.chdir(self.out_path)

        # Ensure the ``RawBundle`` group exists.
        raw_grp = make_group(root, "RawBundle")

        # Attach the external link as a scalar dataset.
        self.zip_link.attach(raw_grp, "raw_bundle.zip")

    @classmethod
    def from_hdf5(cls, root: h5py.File) -> "RawBundleGroup":
        """Restore a :class:`RawBundleGroup` instance from an HDF5 file.

        The method expects a group named ``RawBundle`` containing a scalar
        dataset ``raw_bundle.zip`` that stores the external‑link information.

        Args:
            root: An open :class:`h5py.File` object representing the HDF5 file to
                read from.

        Returns:
            An instance of :class:`RawBundleGroup` populated with the extracted
            :class:`ExternalLink`.

        Raises:
            KeyError: If the ``RawBundle`` group or the ``raw_bundle.zip`` dataset
                does not exist in ``root``.
            TypeError: If the dataset cannot be interpreted as an external link.
            h5py.HDF5Error: Propagated from the underlying HDF5 library when reading
                the dataset fails.
        """
        raw_grp = root["RawBundle"]
        dset = raw_grp["raw_bundle.zip"]
        zip_link = ExternalLink.from_dataset(dset)
        return cls(zip_link=zip_link)
