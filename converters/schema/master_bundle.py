# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0
#
"""master_bundle.py
===================

High‑level container for HDF5 conversion bundles.

The module defines :class:`MasterBundle`, a container that aggregates the three
primary groups produced by ``convert_any_to_hdf5``:

* :class:`~converters.schema.standard_format_group.StandardFormatGroup` – the
  scientific data in the standard format.
* :class:`~converters.schema.raw_bundle_group.RawBundleGroup` – the original
  input data packaged as a zip archive (optional).
* :class:`~converters.schema.protocol_group.ProtocolGroup` – optional protocol
  information packaged as a PDF or other binary document.

Instances can be written to or read from an HDF5 file that follows schema
version ``2.0``.  The implementation supports both the legacy dotted‑string
version attribute (``format_version``) and the newer integer‑based attributes
(``format_version_major``, ``format_version_minor``, ``format_version_patch``).

Typical usage::

    from pathlib import Path
    bundle = MasterBundle()
    bundle.to_hdf5(Path("my_bundle.h5"))

    loaded = MasterBundle.from_hdf5(Path("my_bundle.h5"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import h5py

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.schema.protocol_group import ProtocolGroup
from converters.schema.raw_bundle_group import RawBundleGroup
from converters.schema.standard_format_group import StandardFormatGroup


# Helper – version handling
def _parse_version_string(version: str) -> Tuple[int, int, int]:
    """Parse a dotted version string into a ``(major, minor, patch)`` tuple.

    Missing components are interpreted as ``0`` – this mirrors the behaviour of
    many packaging tools and keeps the function safe for legacy files that only
    provide ``"2.0"``.

    Args:
        version: Dotted version string (e.g. ``"2.0.1"`` or ``"2.0"``).

    Returns:
        A tuple ``(major, minor, patch)`` where each element is an ``int``.

    Raises:
        ValueError: If any component cannot be converted to an integer.
    """
    parts = version.split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return major, minor, patch


# MasterBundle – the public container
@dataclass
class MasterBundle(HDF5Serializable):
    """High‑level container aggregating the groups created by
    ``convert_any_to_hdf5``.

    The container stores:

    * **StandardFormatGroup** – scientific data in the standard format.
    * **RawBundleGroup** – the original input data packaged as a zip archive
      (optional).
    * **ProtocolGroup** – optional protocol information packaged as a PDF or
      other binary document.

    Version handling
    ----------------
    * ``format_version`` (string) – legacy attribute kept for backward
      compatibility.
    * ``format_version_major``, ``format_version_minor``, ``format_version_patch``
      (integers) – the new, machine‑readable representation.

    The integer attributes are added as a one‑liner change to the dataclass,
    incurring virtually no overhead.

    Attributes:
        format_version_major: Major version component (default ``2``).
        format_version_minor: Minor version component (default ``0``).
        format_version_patch: Patch version component (default ``0``).
        standard: Instance of :class:`StandardFormatGroup`.
        protocol: Optional :class:`ProtocolGroup` instance.
        raw: Optional :class:`RawBundleGroup` instance.
    """

    # New integer version fields
    format_version_major: int = 2
    format_version_minor: int = 0
    format_version_patch: int = 0

    # Back‑compatibility string version (legacy)
    @property
    def format_version(self) -> str:
        """Legacy dotted‑string representation of the bundle version.

        Returns:
            A string such as ``'2.0'`` or ``'2.0.1'``.  The patch component is
            omitted when its value is ``0``.
        """
        if self.format_version_patch:
            return f"{self.format_version_major}.{self.format_version_minor}.{self.format_version_patch}"
        return f"{self.format_version_major}.{self.format_version_minor}"

    # The three logical groups
    standard: StandardFormatGroup = field(default_factory=StandardFormatGroup)
    protocol: Optional[ProtocolGroup] = field(default_factory=ProtocolGroup)
    raw: Optional[RawBundleGroup] = field(default_factory=RawBundleGroup)

    # Serialisation – write to HDF5
    def to_hdf5(self, out_path: Path) -> None:
        """Write the bundle to an HDF5 file.

        The method creates the parent directory for ``out_path`` if it does not
        exist, stores version attributes (both legacy and new forms), and
        delegates the actual group serialisation to the contained group objects.

        Args:
            out_path: Destination path for the HDF5 file.

        Raises:
            OSError: If the file cannot be created or written to.
            PermissionError: If the process lacks permission to write to the
                target location.
            h5py.HDF5Error: Propagated from the ``h5py`` library when HDF5‑specific
                errors occur (e.g., corrupted file structures).
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as hf:
            # legacy string attribute
            hf.attrs["format_version"] = self.format_version

            # new integer version attributes
            hf.attrs["format_version_major"] = self.format_version_major
            hf.attrs["format_version_minor"] = self.format_version_minor
            if self.format_version_patch:
                hf.attrs["format_version_patch"] = self.format_version_patch

            # delegate group serialisation
            self.standard.to_hdf5(hf)
            if self.raw:
                self.raw.to_hdf5(hf)
            if self.protocol:
                self.protocol.to_hdf5(hf)

    # Deserialisation – read from HDF5
    @classmethod
    def from_hdf5(cls, in_path: Path) -> "MasterBundle":
        """Create a :class:`MasterBundle` instance from an HDF5 file.

        The method reads version attributes (preferring the integer form but
        falling back to the legacy string attribute) and reconstructs the three
        logical groups if they are present in the file.

        Args:
            in_path: Path to the source HDF5 file.

        Returns:
            A fully populated :class:`MasterBundle` object.

        Raises:
            FileNotFoundError: If ``in_path`` does not exist.
            OSError: If the file cannot be opened for reading.
            h5py.HDF5Error: Propagated from ``h5py`` when reading fails because of
                an invalid or corrupted HDF5 structure.
            ValueError: If the legacy version string cannot be parsed into
                integers.
        """
        with h5py.File(in_path, "r") as hf:
            # Load version – try integer attrs first, then the string attr
            major = hf.attrs.get("format_version_major")
            minor = hf.attrs.get("format_version_minor")
            patch = hf.attrs.get("format_version_patch", 0)

            if major is None or minor is None:
                # Legacy file – decode the dotted string
                legacy_str = hf.attrs.get("format_version", "0.0")
                major, minor, patch = _parse_version_string(str(legacy_str))

            # Load the three logical groups
            std = StandardFormatGroup.from_hdf5(hf)
            raw = RawBundleGroup.from_hdf5(hf) if "RawBundle" in hf else None
            proto = ProtocolGroup.from_hdf5(hf) if "Protocol" in hf else None

            return cls(
                standard=std,
                raw=raw,
                protocol=proto,
                format_version_major=int(major),
                format_version_minor=int(minor),
                format_version_patch=int(patch),
            )