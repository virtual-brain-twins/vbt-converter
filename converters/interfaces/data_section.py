import abc
from pathlib import Path

import h5py
import numpy as np

from converters.utils.data_section_helper import make_dataset


class DataSection(abc.ABC):
    """
    Small building block that knows how to

    * turn an in‑memory representation into an HDF5 group, and
    * read that group back into the in‑memory representation.

    Concrete sections have *no* hard‑wired attribute names – they decide
    what belongs where.  The base class gives you two convenience
    methods for the most common patterns (binary blobs, tables, or raw
    NumPy arrays).
    """

    # Public API
    @abc.abstractmethod
    def to_hdf5(self, grp: h5py.Group) -> None:
        """Write the whole section into the supplied ``grp``."""
        ...

    @abc.abstractmethod
    def from_hdf5(self, grp: h5py.Group) -> None:
        """Populate the section from an existing ``grp``."""
        ...

    # Helper methods – you can call them from your concrete subclass
    @staticmethod
    def _store_numpy_array(
            grp: h5py.Group,
            name: str,
            arr: np.ndarray,
            *,
            compress: bool = True,
    ) -> None:
        """Thin wrapper around ``_make_dataset`` that also stores a tiny attr."""
        dset = make_dataset(grp, name, arr, compress=compress)
        dset.attrs["dtype"] = str(arr.dtype)

    @staticmethod
    def _store_text_blob(
            grp: h5py.Group,
            name: str,
            text: str,
            *,
            compress: bool = True,
    ) -> None:
        """Store a UTF‑8 string as a variable‑length dataset."""
        vlen = h5py.string_dtype(encoding="utf-8")
        dset = grp.create_dataset(name, data=np.array([text], dtype=vlen), compression="gzip" if compress else None)
        dset.attrs["is_vlen"] = True

    @staticmethod
    def _store_binary_blob(
            grp: h5py.Group,
            name: str,
            blob: bytes,
            *,
            compress: bool = True,
    ) -> None:
        arr = np.frombuffer(blob, dtype=np.uint8)
        make_dataset(grp, name, arr, compress=compress)

    @staticmethod
    def read_binary_blob(dset: h5py.Dataset) -> bytes:
        """Read a binary (uint8) dataset back to ``bytes``."""
        return dset[()].tobytes()

    @staticmethod
    def store_external_path(
            grp: h5py.Group,
            name: str,
            path: Path,
            *,
            make_link: bool = True,
    ) -> None:
        """
        Store a reference to a file that lives on the filesystem.

        Parameters
        ----------
        grp : h5py.Group
            The group in which the reference will be stored.
        name : str
            Name of the reference object.
        path : pathlib.Path
            Path to the external file (relative to the HDF5 file's parent directory
            is recommended).
        make_link : bool, optional
            If True **and** ``path`` points to an HDF5 file, create an HDF5 external
            link.  If False (or ``path`` is not an HDF5 file) a plain text dataset
            is created instead.
        """

        # Helper that decides whether we are dealing with a *real* HDF5 file
        def _is_hdf5_file(p: Path) -> bool:
            if not p.is_file():
                return False
            try:
                with h5py.File(p, "r"):
                    return True
            except (OSError, IOError):
                return False

        # When we are allowed to make a link and the target is an HDF5 file
        if make_link and _is_hdf5_file(path):
            # ExternalLink stores only the file name (or absolute path). The
            # object name inside the file is supplied as the second argument.
            # We link to the root group of the target file (`/`).
            external_link = h5py.ExternalLink(str(path), "/")
            grp[name] = external_link
            # NOTE: you cannot attach attributes to the link itself.
            # If you need to keep a human‑readable copy of the path, store it
            # somewhere else, e.g. as an attribute of the parent group.
            grp[name].attrs["link_type"] = "external_hdf5"
            # (optionally) also store the path as a separate attribute for easier
            # inspection – this works because attributes belong to the parent
            # group, not the link object.
            grp.attrs[f"{name}_target"] = str(path)
            return

        # Fallback: store the path as a tiny variable‑length string dataset
        # (a) always keep a human‑readable attribute
        grp.attrs[f"{name}_path"] = str(path)

        # (b) also create a tiny dataset – shape (1,) → not scalar → safe
        dt = h5py.string_dtype(encoding="utf-8")
        data = np.array([str(path)], dtype=dt)
        dset = grp.create_dataset(name, data=data, dtype=dt, compression=None)
        dset.attrs["reference_type"] = "plain_path"
