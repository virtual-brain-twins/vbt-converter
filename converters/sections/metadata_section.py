import json
from typing import Mapping, Dict, Any

import h5py
import numpy as np

from converters.interfaces.data_section import DataSection


class MetadataSection(DataSection):
    """
    Holds any number of key/value pairs that are JSON‑serialisable.
    In the HDF5 file they are stored as **attributes** on the group.
    """
    def __init__(self, data: Mapping[str, Any] | None = None):
        self.data: Dict[str, Any] = dict(data or {})

    def to_hdf5(self, grp: h5py.Group) -> None:
        for k, v in self.data.items():
            # Prefer native HDF5 scalar types; fallback to JSON string.
            if isinstance(v, (int, float, str, bytes, np.number)):
                grp.attrs[k] = v
            else:
                # store JSON‑encoded string – easy to retrieve later
                grp.attrs[k] = json.dumps(v)

    @classmethod
    def from_group(cls, grp: h5py.Group) -> "MetadataSection":
        out = {}
        for k, v in grp.attrs.items():
            try:
                # If it looks like JSON, decode it, otherwise keep original type
                decoded = json.loads(v)
                out[k] = decoded
            except (json.JSONDecodeError, TypeError):
                out[k] = v
        return cls(out)

    def from_hdf5(self, grp: h5py.Group) -> None:
        # Not used directly – we construct via ``from_group``.
        self.data = self.from_group(grp).data