"""converters.schema.protocol_group

This module provides the :class:`ProtocolGroup` data class, which implements the
:class:`~converters.schema.hdf5_serializable.HDF5Serializable` interface for
storing a protocol file (e.g., a binary description of a communication protocol)
inside an HDF5 container.  The class can either embed the raw content directly
or keep a reference to an external file, and it knows how to serialize itself
to and from an HDF5 file.

Typical usage
-------------

>>> from pathlib import Path
>>> pg = ProtocolGroup(version="1.0", content_path=Path("my_protocol.bin"))
>>> pg.to_hdf5(h5_file)          # Write to an open ``h5py.File`` instance
>>> loaded = ProtocolGroup.from_hdf5(h5_file)  # Read back
>>> assert loaded.content == pg.content
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from converters.schema.hdf5_serializable import HDF5Serializable
from converters.utils.dataset_helpers import make_group


@dataclass(slots=True)
class ProtocolGroup(HDF5Serializable):
    """
    Container for a protocol definition that can be stored in an HDF5 file.

    Attributes
    ----------
    version : str
        The protocol version string.  Defaults to ``"1.0"``.
    content_path : Optional[Path]
        Path to the external protocol file.  If ``None`` or the file does not
        exist, the instance must contain embedded content.
    _embedded_content : Optional[bytes]
        Private store for protocol data that has been loaded from an HDF5 file.
        It is populated by :meth:`from_hdf5` and accessed via the
        :attr:`content` property.
    """

    version: str = "1.0"
    content_path: Optional[Path] = None

    _embedded_content: Optional[bytes] = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def content(self) -> Optional[bytes]:
        """
        Return the raw protocol content as ``bytes``.

        The method first checks whether the content has been embedded (i.e.
        loaded from an HDF5 file).  If not, it attempts to read the file pointed
        to by :attr:`content_path`.  If neither source yields data, ``None`` is
        returned.

        Returns
        -------
        Optional[bytes]
            The protocol data, or ``None`` if no data is available.
        """
        if self._embedded_content is not None:
            return self._embedded_content

        if self.content_path and self.content_path.is_file():
            return self.content_path.read_bytes()

        return None

    def to_hdf5(self, root: h5py.File) -> None:
        """
        Serialize the protocol into an HDF5 file.

        The protocol is stored under a group named ``"Protocol"`` with an
        attribute ``Version`` and a dataset ``"Content"`` containing the raw
        bytes.  The dataset is compressed with gzip, shuffled, and chunked.

        Parameters
        ----------
        root : h5py.File
            An open HDF5 file (or group) that will receive the serialized data.

        Raises
        ------
        ValueError
            If :attr:`content_path` is unset or does not point to an existing
            file.
        """
        if not self.content_path or not self.content_path.is_file():
            raise ValueError(
                f"Invalid protocol file: {self.content_path}"
            )

        grp = make_group(root, "Protocol")
        grp.attrs["Version"] = self.version

        grp.create_dataset(
            "Content",
            data=np.fromfile(self.content_path, dtype=np.uint8),
            compression="gzip",
            shuffle=True,
            chunks=True,
        )

    @classmethod
    def from_hdf5(cls, root: h5py.File) -> "ProtocolGroup":
        """
        Deserialize a :class:`ProtocolGroup` from an HDF5 file.

        The method reads the ``"Protocol"`` group, extracts the version attribute,
        and loads the embedded content from the ``"Content"`` dataset.

        Parameters
        ----------
        root : h5py.File
            An open HDF5 file (or group) containing a ``"Protocol"`` group.

        Returns
        -------
        ProtocolGroup
            An instance populated with the data read from the file.
        """
        grp = root["Protocol"]

        obj = cls(version=str(grp.attrs["Version"]))
        obj._embedded_content = grp["Content"][()].tobytes()

        return obj

    # Debug helpers
    def __repr__(self) -> str:
        """
        Return a concise representation useful for debugging.

        The representation includes the version and the source of the protocol
        content (either a file path or ``None``).
        """
        src = (
            f"content_path={self.content_path!s}"
            if self.content_path
            else "content_path=None"
        )
        return f"<ProtocolGroup version={self.version!r} {src}>"