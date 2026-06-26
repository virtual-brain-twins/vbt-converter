import pathlib
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import h5py
import numpy as np

from converters.interfaces.data_section import DataSection
from converters.utils.common import configure_logging

logger = configure_logging(
    __name__,
    log_file_name=Path(__file__).with_suffix(".log").name,
    level=__import__("logging").INFO,
)


class OutputSpikesSection(DataSection):
    """
    A *read‑only* section that mirrors the layout of a Snudda
    ``output_spikes.hdf5`` file (the one you listed with ``h5ls -r``).

    The class is deliberately **lazy** – it does **not** copy the whole
    file into memory unless you explicitly ask for a dataset.  When the
    section is written back into a bundle the original binary layout is
    copied directly on‑disk via ``h5py.Group.copy`` – this is the fastest
    possible way to serialise the data.

    Public helper methods are provided for the most common queries:

    * ``metadata()`` → ``dict`` of all scalar/meta datasets;
    * ``time()``     → 1‑D ``np.ndarray`` (seconds);
    * ``neuron_ids()`` → sorted ``list[int]``;
    * ``spike_train(neuron_id)`` → ``np.ndarray`` of spike times for a given neuron;
    * ``voltage(neuron_id)``    → ``np.ndarray`` of voltage trace;
    * ``raw_dataset(neuron_id, name)`` → any other dataset under a neuron group
      (e.g. ``PKAc`` or modulation‑factor tables).
    """

    def __init__(self, src_file: Optional[Union[str, Path]] = None):
        """
        Parameters
        ----------
        src_file : str | pathlib.Path | None
            Path to the original ``output_spikes.hdf5`` produced by the
            simulation.  If ``None`` the section can still be *re‑created*
            from a bundle (``from_hdf5`` will populate an internal reference).
        """
        self._src_path: Optional[Path] = Path(src_file) if src_file else None
        # When the section is loaded from a bundle we keep a *live* reference
        # to the HDF5 group so that every accessor works on‑demand.
        self._group: Optional[h5py.Group] = None
        # Cached lazy attributes – populated on first read.
        self._metadata_cache: Optional[Dict[str, np.ndarray]] = None
        self._time_cache: Optional[np.ndarray] = None
        self._neuron_ids_cache: Optional[List[int]] = None

    # Internal helpers
    @staticmethod
    def _ensure_file(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Output‑spikes file not found: {path}")

    def _open_src(self) -> h5py.File:
        """Open the source file *once* (read‑only, SWMR)."""
        if self._src_path is None:
            raise RuntimeError(
                "OutputSpikesSection was instantiated without a source file "
                "and has not been populated via `from_hdf5`."
            )
        self._ensure_file(self._src_path)
        # ``libver='latest'`` gives us the newest HDF5 features (fast chunk cache).
        # ``swmr=True`` lets the reader coexist with a writer if the simulation is still running.
        return h5py.File(self._src_path, "r", libver="latest", swmr=True)

    # Public API – lazy read helpers
    def _populate_caches(self) -> None:
        """Read the minimal amount of data needed for the cheap helper methods."""
        if self._group is None:
            raise RuntimeError("Section not attached to a group – call `from_hdf5` first.")
        # 1) metadata
        meta_grp = self._group["meta_data"]
        self._metadata_cache = {k: dset[...] for k, dset in meta_grp.items()}
        # 2) global time vector
        self._time_cache = self._group["time"][...]
        # 3) neuron ids
        self._neuron_ids_cache = sorted(int(k) for k in self._group["neurons"].keys())

    # User‑facing read methods
    def metadata(self) -> Dict[str, np.ndarray]:
        """Return a shallow copy of all ``/meta_data/*`` datasets."""
        if self._metadata_cache is None:
            self._populate_caches()
        # ``dict`` copy to prevent accidental mutation of the internal cache.
        return {k: v.copy() for k, v in self._metadata_cache.items()}

    def time(self) -> np.ndarray:
        """Return the global ``/time`` vector (seconds)."""
        if self._time_cache is None:
            self._populate_caches()
        return self._time_cache.copy()

    def neuron_ids(self) -> List[int]:
        """Sorted list of neuron identifiers present in the file."""
        if self._neuron_ids_cache is None:
            self._populate_caches()
        return list(self._neuron_ids_cache)

    def _neuron_group(self, neuron_id: int) -> h5py.Group:
        """Internal helper that resolves ``/neurons/<id>``."""
        if self._group is None:
            raise RuntimeError("Section not attached to a group – call `from_hdf5` first.")
        try:
            return self._group["neurons"][str(neuron_id)]
        except KeyError as exc:
            raise KeyError(f"Neuron id {neuron_id} not present in the output file") from exc

    def spike_train(self, neuron_id: int) -> np.ndarray:
        """
        Return a **1‑D** array with the spike times (seconds) of ``neuron_id``.
        The original layout is ``[1, N]`` – we squeeze the leading singleton.
        """
        grp = self._neuron_group(neuron_id)
        return grp["spikes"][...].squeeze()

    def voltage(self, neuron_id: int) -> np.ndarray:
        """Return the voltage trace (shape ``[1, T]`` → squeezed to 1‑D)."""
        grp = self._neuron_group(neuron_id)
        return grp["voltage"][...].squeeze()

    def raw_dataset(self, neuron_id: int, name: str) -> np.ndarray:
        """
        Access *any* other dataset stored under a neuron group
        (e.g. ``PKAc`` or ``kas_ms.modulation_factor``).
        """
        grp = self._neuron_group(neuron_id)
        if name not in grp:
            raise KeyError(f"Dataset '{name}' not found for neuron {neuron_id}")
        return grp[name][...]

    # DataSection implementation
    def to_hdf5(self, grp: h5py.Group) -> None:
        """
        Write the whole section into ``grp`` **without** materialising the data.

        The algorithm is:
        1. Open the original file (if we have a path).
        2. Use ``h5py.Group.copy`` for the three top‑level objects
           ``meta_data``, ``time`` and ``neurons``.
        3. Preserve the original compression/chunking settings automatically.

        If a source file was not supplied (e.g. we built the section manually)
        we fall back to a slower, but still correct, NumPy‑based copy.
        """
        if self._src_path is not None:
            # Fast path – direct HDF5 copy
            with self._open_src() as src:
                for obj_name in ("meta_data", "time", "neurons"):
                    if obj_name not in src:
                        logger.warning(
                            "Source file %s is missing expected top‑level object %s",
                            self._src_path,
                            obj_name,
                        )
                        continue
                    src.copy(obj_name, grp)          # <-- one C call per object
        else:
            # Slow – we have nothing to copy from, we must rely on the cached
            # group (populated via ``from_hdf5``).  We copy element‑wise using the
            # helper methods from ``DataSection``.
            if self._group is None:
                raise RuntimeError(
                    "OutputSpikesSection has neither a source file nor a "
                    "populated group – cannot serialise."
                )
            # meta_data
            meta_grp = grp.create_group("meta_data")
            for name, dset in self._group["meta_data"].items():
                DataSection._store_numpy_array(meta_grp, name, dset[...])

            # time
            DataSection._store_numpy_array(grp, "time", self._group["time"][...])

            # neurons – copy each neuron sub‑group recursively
            neurons_dst = grp.create_group("neurons")
            for nid_str, src_ngrp in self._group["neurons"].items():
                dst_ngrp = neurons_dst.create_group(nid_str)
                for dname, dset in src_ngrp.items():
                    DataSection._store_numpy_array(dst_ngrp, dname, dset[...])

    def from_hdf5(self, grp: h5py.Group) -> None:
        """
        Attach the section to an *already opened* HDF5 group (the one created
        by ``SimulationBundle.to_hdf5``).  All subsequent accessor calls will read
        lazily from this group – no copy is performed.
        """
        # Keep a *live* reference; the group belongs to the parent HDF5 file which
        # stays open for the whole lifetime of the bundle object.
        self._group = grp
        # Reset cached items – they will be lazily (re)filled on demand.
        self._metadata_cache = None
        self._time_cache = None
        self._neuron_ids_cache = None