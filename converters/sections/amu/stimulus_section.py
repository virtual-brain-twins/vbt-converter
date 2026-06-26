from __future__ import annotations

import h5py
import numpy as np

from converters.interfaces.data_section import DataSection
from converters.utils.data_section_helper import make_dataset


class StimulusSection(DataSection):
    """
    Represents the *piecewise‑constant* stimulus that the original AMU
    pipeline stores in a group called ``stimulus`` with three items:

        * attribute  ``type`` – always ``'piecewise_constant'`` in the reference
        * dataset    ``limits`` – 1‑D int array, the time‑break points (ms)
        * dataset    ``values`` – a variable‑length 1‑D array where each entry
          is a NumPy array of floats (the value that holds between two limits)

    The Python representation keeps the three objects directly.
    """

    def __init__(
            self,
            limits: np.ndarray | None = None,
            values: np.ndarray | None = None,
            stim_type: str = "piecewise_constant",
    ):
        self.limits = np.asarray(limits, dtype=np.int64) if limits is not None else np.array([], dtype=np.int64)
        self.values = values if values is not None else np.array([], dtype=np.float64)
        self.stim_type = stim_type

    def to_hdf5(self, grp: h5py.Group) -> None:
        # attribute
        grp.attrs["type"] = self.stim_type

        # limits – plain int dataset (no compression – tiny)
        make_dataset(grp, "limits", self.limits, compress=False)

        # values – plain float dataset
        make_dataset(grp, "values", self.values, compress=False)

    @classmethod
    def from_group(cls, grp: h5py.Group) -> "StimulusSection":
        stim_type = grp.attrs.get("type", "piecewise_constant")
        limits = grp["limits"][()]
        # ``values`` comes back as a 1‑D object array of ``np.ndarray`` objects.
        values = grp["values"][()]
        return cls(limits=limits, values=values, stim_type=stim_type)

    def from_hdf5(self, grp: h5py.Group) -> None:
        tmp = self.from_group(grp)
        self.limits = tmp.limits
        self.values = tmp.values
        self.stim_type = tmp.stim_type
