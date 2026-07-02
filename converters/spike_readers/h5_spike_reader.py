import logging
import pathlib
from typing import Union, Optional, Tuple

import h5py
import numpy as np

from converters.spike_readers.base_spike_reader import BaseSpikeReader
from converters.utils.common import configure_logging

log = configure_logging(
    __name__,
    log_file_name=pathlib.Path(__file__).with_suffix(".log").name,
    level=logging.INFO,
)


class H5SpikeReader(BaseSpikeReader):
    """
    Reads a spike‑file that can be in **one of two layouts**:

    1. *Flat* layout (the classic Snudda ``/spikes`` group):
       ``/spikes/indices`` (optional) and ``/spikes/times`` (mandatory).

    2. *Per‑neuron* layout – the one you posted – where each neuron lives in
       ``/neurons/<gid>/spikes`` (shape ``(1, N_spikes)``).  In this case the
       optional ``idx_dset``/``time_dset`` arguments are ignored and the
       reader automatically flattens the hierarchy into ``(indices, times)``.

    The returned arrays are **lazy** `h5py.Dataset` objects (or a view on them)
    so no full copy is performed unless the caller slices the data.
    """

    def __init__(
            self,
            file_path: Union[str, pathlib.Path],
            *,
            idx_dset: Optional[str] = "/spikes/indices",
            time_dset: str = "/spikes/times",
    ):
        self.file_path = pathlib.Path(file_path)
        self.idx_dset = idx_dset
        self.time_dset = time_dset

    # Private helpers
    def _open(self):
        """Open the file for read‑only, fast‑cache SWMR access."""
        return h5py.File(self.file_path, "r", libver="latest", swmr=True)

    @staticmethod
    def _flatten_per_neuron_group(root: h5py.Group) -> Tuple[np.ndarray, np.ndarray]:
        """
        Turn the ``/neurons/<gid>/spikes`` hierarchy into two arrays:

        * ``indices`` – repeated gid for each spike,
        * ``times``   – concatenated spike times.

        The function performs *only one* allocation for each array.
        """
        neurons = root["neurons"]
        # First pass – total number of spikes
        total_spikes = 0
        gid_list = []
        for gid_str, grp in neurons.items():
            dset = grp["spikes"]
            n = dset.shape[1]  # shape is (1, N_spikes)
            total_spikes += n
            gid_list.append((int(gid_str), n))

        # Allocate contiguous arrays
        indices = np.empty(total_spikes, dtype=np.int64)
        times = np.empty(total_spikes, dtype=np.float64)

        # Second pass – copy data directly into the pre‑allocated buffers
        pos = 0
        for gid, n in gid_list:
            dset = neurons[str(gid)]["spikes"]
            # ``read_direct`` copies straight into the target slice without an
            # intermediate Python object.
            dset.read_direct(times[pos: pos + n])
            indices[pos: pos + n] = gid
            pos += n

        return indices, times

    # Public API
    def get_spike_arrays(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Return the spike arrays *lazily*:

        * ``indices`` – ``h5py.Dataset`` (or ``None``) with ``int64`` dtype,
        * ``times``   – ``h5py.Dataset`` with ``float64`` dtype.

        If the file follows the per‑neuron layout, the method *flattens* the
        hierarchy into two **in‑memory** arrays – this is still very fast because
        the data are read only once and stored in a single contiguous block.
        """
        if not self.file_path.is_file():
            log.error("HDF5 file %s does not exist – returning empty arrays", self.file_path)
            return None, np.empty(0, dtype=float)

        with self._open() as f:
            # Detect which layout we have.
            if "neurons" in f:  # per‑neuron layout
                log.debug("Detected per‑neuron spike layout in %s", self.file_path)
                return self._flatten_per_neuron_group(f)

            # Classic flat layout – just return the datasets (no copy).
            if self.time_dset not in f:
                raise KeyError(
                    f"Dataset '{self.time_dset}' not found in {self.file_path!s}. "
                    f"Available top‑level keys: {list(f.keys())}"
                )
            times = f[self.time_dset]  # h5py.Dataset – lazy

            if self.idx_dset is None:
                indices = None
            else:
                if self.idx_dset not in f:
                    log.warning(
                        "Indices dataset %s not present – returning None", self.idx_dset
                    )
                    indices = None
                else:
                    indices = f[self.idx_dset]  # also lazy

            return indices, times
