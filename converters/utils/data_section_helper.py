"""Utility helpers for working with HDF5 files via h5py.

This module provides small convenience functions that are used when creating HDF5
datasets:

* ``_ensure_parent`` – guarantees that the parent directory of a supplied
  :class:`pathlib.Path` exists.
* ``_compressible_dtype`` – heuristically decides whether a NumPy array is a good
  candidate for gzip compression.
* ``make_dataset`` – creates a dataset in an existing
  :class:`h5py.Group`, automatically selecting compression and chunking options
  based on the data type and size.

These helpers are deliberately lightweight and have no external side‑effects
apart from creating directories on the filesystem and writing to the supplied
HDF5 group.
"""

from pathlib import Path
from typing import Union, Optional, Tuple, MutableMapping, Any

import h5py
import numpy as np


def _ensure_parent(p: Path) -> None:
    """Create the parent directory of *p* if it does not already exist.

    The function is tolerant of race conditions and does nothing if the
    directory hierarchy is already present.

    Args:
        p: A :class:`pathlib.Path` object whose ``parent`` directory should be
           ensured.

    Raises:
        OSError: If the directory cannot be created due to permission errors,
                 missing intermediate components, or other OS‑level problems.
    """
    p.parent.mkdir(parents=True, exist_ok=True)


def _compressible_dtype(arr: np.ndarray) -> bool:
    """Return ``True`` if *arr* is a good candidate for gzip compression.

    Float, signed and unsigned integer arrays larger than 1 KiB usually
    compress well, while string/bytes arrays rarely benefit from gzip.

    Args:
        arr: A NumPy ``ndarray`` to inspect.

    Returns:
        ``True`` if the array's dtype is numeric (float, int, or unsigned int) and
        the total size exceeds 1 KiB; otherwise ``False``.

    Raises:
        TypeError: If *arr* cannot be interpreted as a NumPy ``ndarray``.
    """
    # Float / int arrays of moderate size compress nicely; string arrays usually do not.
    return arr.dtype.kind in {"f", "i", "u"} and arr.nbytes > 1024


def make_dataset(
    grp: h5py.Group,
    name: str,
    data: Union[np.ndarray, str, bytes],
    *,
    compress: bool = True,
    chunks: Optional[Tuple[int, ...]] = None,
) -> h5py.Dataset:
    """Create a dataset in *grp* with sensible defaults for compression and chunking.

    The function automatically decides whether to apply gzip compression based on
    the data type and size, and it picks a chunk shape that targets roughly 1 MiB
    per chunk for large numeric arrays.  Callers can override the behaviour by
    passing explicit ``compress`` or ``chunks`` arguments.

    Args:
        grp: An open :class:`h5py.Group` (or ``File``) that will contain the new
            dataset.
        name: Name of the dataset to create within ``grp``.
        data: Data to store – can be a NumPy array, a string, or raw bytes.
        compress: If ``True`` (default), attempt to gzip‑compress the dataset when
            the data is deemed compressible.
        chunks: Optional explicit chunk shape.  If omitted and compression is
            enabled for a large array, a heuristic 1 MiB chunk size is used.

    Returns:
        The newly created :class:`h5py.Dataset` instance.

    Raises:
        TypeError: If *data* is of an unsupported type or cannot be converted to a
            NumPy array when compression heuristics are applied.
        ValueError: If *chunks* is supplied but is incompatible with the shape of
            *data*.
        RuntimeError: If the underlying HDF5 library is unable to create the
            dataset (e.g., due to insufficient space, name conflicts, or group
            write permissions).
        OSError: Propagated from filesystem‑related errors when underlying HDF5
            operations interact with the file system.
    """
    kw: MutableMapping[str, Any] = {}
    if compress and _compressible_dtype(np.asarray(data)):
        kw["compression"] = "gzip"
        kw["compression_opts"] = 6
        # If the caller did not give a chunk size, use a ~1MiB chunk for large arrays.
        if chunks is None and isinstance(data, np.ndarray) and data.nbytes > 4 * 1024 * 1024:
            # simple heuristic: split the first dimension into ~1MiB chunks
            elem_bytes = data.dtype.itemsize
            dim0 = max(1, (1 * 1024 * 1024) // (elem_bytes or 1))
            chunks = (dim0,) + data.shape[1:]
        kw["chunks"] = chunks
    return grp.create_dataset(name, data=data, **kw)