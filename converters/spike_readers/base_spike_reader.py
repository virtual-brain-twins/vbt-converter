"""
Minimal interface for a spike‑storage backend.

The two arrays returned are:

* ``indices`` – integer neuron IDs (or ``None`` if the file does not store them);
* ``times``   – float times (seconds) of the corresponding spikes.

Both arrays may be *memory‑mapped* (i.e. ``h5py.Dataset`` objects) and therefore
are virtually zero‑cost to obtain.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np


class BaseSpikeReader(ABC):
    @abstractmethod
    def get_spike_arrays(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Return ``(indices, times)`` – both may be ``h5py.Dataset`` objects."""
        raise NotImplementedError