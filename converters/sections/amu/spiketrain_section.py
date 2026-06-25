from typing import Mapping, Tuple, Dict

import h5py
import numpy as np

from converters.interfaces.data_section import DataSection
from converters.utils.data_section_helper import make_dataset


class SpikeTrainSection(DataSection):
    """
    Stores spike‐times for any number of populations.
    Internal layout:

    /<section>/population_A/
        indices : int[N]   (optional, for indexed storage)
        times   : float[N] (seconds)
    """

    def __init__(self, data: Mapping[str, Tuple[np.ndarray, np.ndarray]] | None = None,
                 max_inline_bytes: int = 200 * 1024 * 1024):
        """
        ``data`` maps *population name* → (indices, times). ``indices`` may be
        ``None`` if you only want the plain ``times`` array.
        """
        self.max_inline_bytes = max_inline_bytes  # <-- decides embed vs link
        self.data: Dict[str, Tuple[np.ndarray | None, np.ndarray]] = dict(data or {})

    # ------------------------------------------------------------------
    def to_hdf5(self, grp: h5py.Group) -> None:
        for pop, (idx, times) in self.data.items():
            pop_grp = grp.create_group(pop)
            if idx is not None:
                make_dataset(pop_grp, "indices", idx.astype(np.int64), compress=False)
            make_dataset(pop_grp, "times", times.astype(np.float64), compress=True)

    # ------------------------------------------------------------------
    @classmethod
    def from_group(cls, grp: h5py.Group) -> "SpikeTrainSection":
        out = {}
        for pop_name, pop_grp in grp.items():
            idx = pop_grp["indices"][()] if "indices" in pop_grp else None
            times = pop_grp["times"][()]
            out[pop_name] = {"indices": idx, "times": times}
        return cls(out)

    def from_hdf5(self, grp: h5py.Group) -> None:
        self.data = self.from_group(grp).data
