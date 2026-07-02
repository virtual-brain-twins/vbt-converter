import pathlib
from typing import Union, Optional

from converters.spike_readers.base_spike_reader import BaseSpikeReader
from converters.spike_readers.h5_spike_reader import H5SpikeReader
from converters.spike_readers.npz_spike_reader import NpzSpikeReader


def make_spike_reader(
    file_path: Union[str, pathlib.Path],
    *,
    idx_dset: Optional[str] = None,
    time_dset: Optional[str] = None,
) -> BaseSpikeReader:
    """
    Return an instance of the appropriate ``BaseSpikeReader`` subclass
    for *file_path*.

    Parameters
    ----------
    file_path : Union[str, pathlib.Path]
        Path to the spike file.
    idx_dset, time_dset : str, optional
        Forwarded to ``H5SpikeReader`` – ignored for the npz case.

    Returns
    -------
    BaseSpikeReader
        Either ``NpzSpikeReader`` or ``H5SpikeReader``.
    """
    p = pathlib.Path(file_path)
    if p.suffix.lower() == ".npz":
        return NpzSpikeReader(p)
    elif p.suffix.lower() in {".h5", ".hdf5", ".hdf"}:
        return H5SpikeReader(p, idx_dset=idx_dset, time_dset=time_dset or "/spikes/times")
    else:
        raise ValueError(
            f"Unsupported spike file extension '{p.suffix}'.  Expected .npz, .h5, .hdf5."
        )