from __future__ import annotations

from typing import Mapping, Dict

import h5py
import numpy as np

from converters.interfaces.data_section import DataSection
from converters.utils.data_section_helper import make_dataset


class RecordNeuronsSection(DataSection):
    """
    Stores the per‑neuron “record” information that the original AMU
    simulation writes to ``record_info/record_neurons.data``.
    Internally we keep a plain ``dict[str, int]`` mapping *label → value*.
    The on‑disk representation is a **structured** 1‑D array with the fields

        (label : UTF‑8 string, value : int64)
    """

    def __init__(self, data: Mapping[str, int] | None = None):
        # keep a mutable copy
        self.data: Dict[str, int] = dict(data or {})

    # HDF5 write
    def to_hdf5(self, grp: h5py.Group) -> None:
        """
        Create a single dataset called ``record_neurons`` that contains a
        structured array ``[('label', <vlen str>), ('value', int64)]``.
        The dataset is compressed (gzip) – the same choice as the reference
        script.
        """
        # build the structured array
        dtype = np.dtype(
            [("label", h5py.string_dtype(encoding="utf-8")), ("value", np.int64)]
        )
        arr = np.empty(len(self.data), dtype=dtype)
        arr["label"] = list(self.data.keys())
        arr["value"] = list(self.data.values())

        make_dataset(grp, "record_neurons", arr, compress=True)

    # HDF5 read helpers
    @classmethod
    def from_group(cls, grp: h5py.Group) -> "RecordNeuronsSection":
        """
        Re‑create the section from a group that already contains the
        ``record_neurons`` dataset.
        """
        raw = grp["record_neurons"][()]               # structured numpy array
        data = {label.decode(): int(val) for label, val in zip(raw["label"], raw["value"])}
        return cls(data)

    def from_hdf5(self, grp: h5py.Group) -> None:
        self.data = self.from_group(grp).data