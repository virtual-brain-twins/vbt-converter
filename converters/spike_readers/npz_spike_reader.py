import pathlib
from typing import Union, Tuple, Optional

import numpy as np

from converters.spike_readers.base_spike_reader import BaseSpikeReader


class NpzSpikeReader(BaseSpikeReader):
    """
    Reads a ``.npz`` file that contains at least a ``times`` array and
    optionally an ``indices`` array.

    Parameters
    ----------
    file_path : Union[str, pathlib.Path]
        Path to the ``.npz`` file.

    Notes
    -----
    ``np.load(..., mmap_mode='r')`` gives a read‑only memory‑mapped view,
    i.e. no data are copied into RAM until you actually access them.
    """

    def __init__(self, file_path: Union[str, pathlib.Path]):
        self.file_path = pathlib.Path(file_path)

    def get_spike_arrays(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        # ``allow_pickle=False`` is safe because we only expect ndarrays.
        with np.load(self.file_path, allow_pickle=False, mmap_mode='r') as npz:
            times = npz["times"]
            # ``dict.get`` works on a ``NpzFile`` as well.
            indices = npz.get("indices", None)
        return indices, times