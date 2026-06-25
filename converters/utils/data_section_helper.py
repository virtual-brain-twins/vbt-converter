# ------------------------------------------------------------------
# Helpers that are used by many sections
# ------------------------------------------------------------------
from pathlib import Path
from typing import Union, Optional, Tuple, MutableMapping, Any

import h5py
import numpy as np


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _compressible_dtype(arr: np.ndarray) -> bool:
    """Return True if the array is a good candidate for gzip compression."""
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
    """Create a dataset, automatically picking compression and chunking."""
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
