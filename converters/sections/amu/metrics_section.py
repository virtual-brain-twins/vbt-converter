from __future__ import annotations

from typing import Mapping, Dict

import h5py
import numpy as np

from converters.interfaces.data_section import DataSection
from converters.utils.data_section_helper import make_dataset


class MetricsSection(DataSection):
    """
    Holds the content of ``metrics/avg_firing_rates.csv``.
    In‑memory we keep a mapping *population label → average firing rate*.
    On‑disk this is stored as a **structured** dataset with the fields

        (label : UTF‑8 string, value : float64)
    """

    def __init__(self, data: Mapping[str, float] | None = None):
        self.data: Dict[str, float] = dict(data or {})

    # ------------------------------------------------------------------
    def to_hdf5(self, grp: h5py.Group) -> None:
        dtype = np.dtype(
            [("label", h5py.string_dtype(encoding="utf-8")), ("value", np.float64)]
        )
        arr = np.empty(len(self.data), dtype=dtype)
        arr["label"] = list(self.data.keys())
        arr["value"] = list(self.data.values())

        # The reference converter stored the dataset under the *group* name
        # ``Avg Firing Rate Hz`` – we keep exactly that name for compatibility.
        make_dataset(grp, "Avg Firing Rate Hz", arr, compress=True)

    # ------------------------------------------------------------------
    @classmethod
    def from_group(cls, grp: h5py.Group) -> "MetricsSection":
        raw = grp["Avg Firing Rate Hz"][()]
        data = {label: float(val) for label, val in zip(raw["label"], raw["value"])}
        return cls(data)

    def from_hdf5(self, grp: h5py.Group) -> None:
        self.data = self.from_group(grp).data
