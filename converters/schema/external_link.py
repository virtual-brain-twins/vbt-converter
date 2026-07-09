"""Utility for storing and retrieving external‑link information in an HDF5 file.

This module defines :class:`ExternalLink`, a small helper that stores a ZIP
*external link* as a **scalar dataset**.  The dataset itself carries no useful
data – it only holds two attributes (``path`` and ``target``) that describe the
linked file and the object inside that file.  The implementation works
with ZIP files because HDF5 attributes can only be attached to real HDF5
objects, not to raw binary data.

Typical usage
--------------

```python
with h5py.File("container.h5", "a") as f:
    grp = f.require_group("links")
    link = ExternalLink(path="data.zip", target="/")
    link.attach(grp, name="my_link")

    # Later, read it back:
    dset = grp["my_link"]
    restored = ExternalLink.from_dataset(dset)
```
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py


@dataclass
class ExternalLink:
    """Wrapper that stores a ZIP *external link* as a scalar HDF5 dataset.

    The dataset contains a single dummy byte; only the attached attributes are
    relevant.

    Attributes:
        path: File‑system path (relative or absolute) of the linked file.
        target: Object path inside the linked file (defaults to the root, ``"/"``).

    Methods:
        attach: Write the link information to a given HDF5 group.
        from_dataset: Re‑construct an :class:`ExternalLink` from a matching dataset.
    """

    path: str
    target: str = "/"  #: Always the root of the linked file.

    def attach(self, grp: h5py.Group, name: str) -> None:
        """Store the link information as a scalar dataset.

        A scalar dataset named ``name`` is created inside ``grp``.  The dataset
        holds a single dummy byte (value ``0``); the important information is
        stored in four attributes:

        * ``path`` – the file system path of the linked file.
        * ``target`` – the object path inside the linked file.
        * ``link_type`` – the constant string ``"external_zip"``.
        * ``format`` – the constant string ``"zip"``.

        Args:
            grp: The HDF5 group in which to create the dataset.
            name: The name of the dataset to create.

        Raises:
            TypeError: If ``grp`` is not an instance of :class:`h5py.Group`.
            ValueError: If ``name`` is an empty string or otherwise invalid.
            OSError: If the underlying HDF5 library fails to create the dataset
                (e.g., due to I/O errors or name collisions).
        """
        if not isinstance(grp, h5py.Group):
            raise TypeError("grp must be an h5py.Group instance")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non‑empty string")

        # Create a **scalar** dataset – size‑0 bytes is not allowed, so we use a
        # tiny integer as a placeholder.
        dset: h5py.Dataset = grp.create_dataset(name, shape=(), dtype="i1")
        dset[()] = 0  # Write the dummy value.

        # Attach the metadata that describes the external link.
        dset.attrs["path"] = self.path
        dset.attrs["target"] = self.target
        dset.attrs["link_type"] = "external_zip"
        dset.attrs["format"] = "zip"

    @classmethod
    def from_dataset(cls, dset: h5py.Dataset) -> "ExternalLink":
        """Re‑construct an :class:`ExternalLink` from a dataset created by :meth:`attach`.

        The method reads the ``path`` and ``target`` attributes from ``dset``.
        Missing attributes fall back to sensible defaults (empty string for
        ``path`` and ``"/"`` for ``target``).

        Args:
            dset: The HDF5 dataset that stores an external‑link scalar dataset.

        Returns:
            An :class:`ExternalLink` instance populated with the retrieved metadata.

        Raises:
            TypeError: If ``dset`` is not an instance of :class:`h5py.Dataset`.
            OSError: If reading attributes fails due to an underlying HDF5 error.
        """
        if not isinstance(dset, h5py.Dataset):
            raise TypeError("dset must be an h5py.Dataset instance")

        path: str = dset.attrs.get("path", "")
        target: str = dset.attrs.get("target", "/")
        return cls(path=path, target=target)
