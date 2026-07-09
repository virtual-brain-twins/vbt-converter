"""Utility functions for creating HDF5 groups and datasets.

This module provides helpers that simplify common HDF5 operations:

* :func:`make_group` creates (or replaces) a child group within a parent
  ``h5py.Group``.
* :func:`store_dataset` stores arbitrary data in an ``h5py.Group`` with sensible
  defaults such as gzip compression and automatic chunk sizing for large
  arrays.

Both functions raise standard HDF5‐related exceptions (e.g. :class:`OSError`)
when underlying operations fail.
"""

from typing import Any, Tuple, Dict

import h5py
import numpy as np


def make_group(parent: h5py.Group, name: str) -> h5py.Group:
    """Create (or replace) a child group within an HDF5 parent group.

    If a group or dataset with the given ``name`` already exists under ``parent``,
    it is deleted before the new group is created.

    Args:
        parent: The HDF5 group that will contain the new child group.
        name: The name of the child group to create.

    Returns:
        The newly created :class:`h5py.Group` instance.

    Raises:
        OSError: If the underlying HDF5 library fails to delete the existing
            member or to create the new group.
        TypeError: If ``parent`` is not an ``h5py.Group`` instance.
    """
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def store_dataset(
    grp: h5py.Group,
    name: str,
    data: np.ndarray | Any,
    *,
    compression: bool = True,
    chunks: Tuple[int, ...] | None = None,
) -> h5py.Dataset:
    """Store data in an HDF5 group as a dataset with sensible defaults.

    This wrapper around :meth:`h5py.Group.create_dataset` adds gzip compression
    (level 6) by default, selects ~1 MiB chunk sizes for large array data, and
    avoids passing filter‑related arguments for scalar datasets (which HDF5 does
    not support).

    Args:
        grp: The HDF5 group in which to create the dataset.
        name: Name of the dataset.
        data: Data to store. Can be a NumPy ``ndarray``, a Python scalar, or a
            sequence (list/tuple) which will be converted to an ``ndarray``.
        compression: Whether to enable gzip compression for non‑scalar datasets.
            Defaults to ``True``.
        chunks: Optional explicit chunk shape. If ``None`` and the data size
            exceeds ~4 MiB, an automatic chunk size of roughly 1 MiB for the
            first dimension is chosen.

    Returns:
        The created :class:`h5py.Dataset` object.

    Raises:
        TypeError: If ``data`` cannot be converted to a ``numpy.ndarray``.
        ValueError: If the provided ``chunks`` are incompatible with the data
            shape.
        OSError: If the underlying HDF5 operation fails (e.g., insufficient
            disk space or permission errors).
    """
    # Normalise ``data`` – we want to treat Python scalars the same way as 0‑D ndarrays.
    if np.isscalar(data):
        # Convert to a 0‑D ndarray so that ``data.shape`` exists.
        # ``astype`` will be a no‑op for an already‑numpy scalar.
        data = np.array(data)
    elif isinstance(data, (list, tuple)):
        # Turn plain sequences into an ndarray; this also guarantees a ``shape`` attribute.
        data = np.asarray(data)

    # Decide whether we are dealing with a scalar dataset.
    is_scalar = data.shape == ()   # ``()`` means “no dimensions”

    # Assemble keyword arguments only when they make sense.
    kw: Dict[str, Any] = {}

    if not is_scalar:  # Only non‑scalar datasets can be filtered
        if compression:
            kw["compression"] = "gzip"
            kw["compression_opts"] = 6

        # Auto‑chunk for “large” arrays (larger than ~4 MiB)
        if isinstance(data, np.ndarray) and chunks is None and data.nbytes > 4 * 1024 * 1024:
            # ~1 MiB chunks for the first dimension
            elem = data.dtype.itemsize or 1
            dim0 = max(1, (1 * 1024 * 1024) // elem)
            kw["chunks"] = (dim0,) + data.shape[1:]

        # Explicit user‑provided chunk specification overrides the auto‑chunk
        if chunks is not None:
            kw["chunks"] = chunks

    # Finally create the dataset.
    return grp.create_dataset(name, data=data, **kw)