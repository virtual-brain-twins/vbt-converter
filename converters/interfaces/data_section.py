import abc
from pathlib import Path

import h5py
import numpy as np

from converters.utils.data_section_helper import _make_dataset


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

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def to_hdf5(self, grp: h5py.Group) -> None:
        """Write the whole section into the supplied ``grp``."""
        ...

    @abc.abstractmethod
    def from_hdf5(self, grp: h5py.Group) -> None:
        """Populate the section from an existing ``grp``."""
        ...

    # ------------------------------------------------------------------
    #  Helper methods – you can call them from your concrete subclass
    # ------------------------------------------------------------------
    @staticmethod
    def _store_numpy_array(
        grp: h5py.Group,
        name: str,
        arr: np.ndarray,
        *,
        compress: bool = True,
    ) -> None:
        """Thin wrapper around ``_make_dataset`` that also stores a tiny attr."""
        dset = _make_dataset(grp, name, arr, compress=compress)
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
        _make_dataset(grp, name, arr, compress=compress)

    @staticmethod
    def _read_binary_blob(dset: h5py.Dataset) -> bytes:
        """Read a binary (uint8) dataset back to ``bytes``."""
        return dset[()].tobytes()

    def _store_external_path(
            grp: h5py.Group,
            name: str,
            path: Path,
            *,
            make_link: bool = True,
    ) -> None:
        """
        Store a *reference* to a file that lives on the filesystem.

        * If ``make_link`` is True, an **HDF5 external link** is created.  The
          linked file can be opened directly with ``h5py.File(linked_path)``.
        * If ``make_link`` is False, we simply store the string path as a
          UTF‑8 scalar dataset (so the reference survives when the HDF5 file is
          moved to a different machine; the user can decide later how to resolve it).

        The function does **not** read the target file at all.
        """
        if make_link:
            # ``target`` is the absolute path against which the link is interpreted.
            # ``source`` is the group that will contain the link.
            # ``external_link`` creates an object that behaves like a dataset but
            # lives in another file.
            external_link = h5py.ExternalLink(str(path), "/")
            # HDF5 stores external links as *soft objects* – they have no data payload.
            grp[name] = external_link
            grp[name].attrs["link_target"] = str(path)  # for humans / JSON dumps
        else:
            # Store the path as a tiny vlen string dataset (still < 1 kB)
            DataSection._store_text_blob(grp, name, str(path), compress=False)