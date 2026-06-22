#  BUNDLE INTERFACE
import abc
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Type

import h5py

from converters.interfaces.data_section import DataSection
from converters.sections.metadata_section import MetadataSection
from converters.utils.common import configure_logging

logger = configure_logging(
    __name__,
    log_file_name=Path(__file__).with_suffix(".log").name,
    level=logging.INFO,
)


class SimulationBundle(abc.ABC):
    """
    Abstract representation of *one* simulation (i.e. one row in the CSV / one
    directory in a sweep).  A bundle owns a dictionary ``sections`` that maps
    a human-readable key → :class:`DataSection` instance.

    The base class implements the plumbing that writes a **single**
    HDF5 file containing:

    * a top‑level group ``/sim_<id>`` (the id is supplied by the concrete class)
    * one subgroup per registered ``DataSection``
    * an optional *raw‑bundle* (a .zip/.tar.gz of the original directory)
    """

    # Required by every concrete subclass
    @property
    @abc.abstractmethod
    def sim_id(self) -> str:
        """Unique identifier that will become the name of the top‑level group."""
        ...

    @property
    @abc.abstractmethod
    def original_root(self) -> Path:
        """Root directory that contains the raw simulation files."""
        ...

    # Optional – can be overridden if custom handling is needed
    def __init__(self) -> None:
        # A mapping *section_name* → DataSection instance
        self.sections: Dict[str, DataSection] = {}

    # Section registration helpers
    def register_section(self, name: str, section: DataSection) -> None:
        """Add a new logical section (e.g. ``'spikes'`` or ``'metadata'``)."""
        if name in self.sections:
            raise KeyError(f"Section '{name}' already registered")
        self.sections[name] = section

    # High‑level (de)serialisation
    def to_hdf5(self, h5file: h5py.File, *, embed_raw_bundle: bool = True) -> None:
        """
        Write the whole bundle into ``h5file`` under ``/sim_<id>``.

        Parameters
        ----------
        embed_raw_bundle: bool
            If ``True`` (default) the original directory is zipped and stored
            as a binary blob named ``raw_bundle``.  Set to ``False`` when you
            want a *pure* metadata‑only file and keep the original data on disk.
        """
        grp = h5file.create_group(f"sim_{self.sim_id}")
        logger.debug("Creating group %s for simulation %s", grp.name, self.sim_id)

        # Write each registered section
        for name, sec in self.sections.items():
            sub = grp.create_group(name)
            logger.debug("→ writing section '%s' (%s)", name, sub.name)
            sec.to_hdf5(sub)

        # Store a compact zip of the raw directory (optional)
        # if embed_raw_bundle:
        #     raw_blob = self._make_raw_bundle_blob()
        #     DataSection._store_binary_blob(
        #         grp,
        #         name="raw_bundle",
        #         blob=raw_blob,
        #         compress=True,
        #     )
        #     grp["raw_bundle"].attrs["format"] = "zip"
        #     grp["raw_bundle"].attrs["original_root"] = str(self.original_root)

    @classmethod
    def from_hdf5(cls: Type["SimulationBundle"], h5file: h5py.File, sim_id: str) -> "SimulationBundle":
        """
        Re‑create a concrete ``SimulationBundle`` from a file.

        The concrete subclass must provide a ``_factory`` class‑method that
        knows how to instantiate itself from the *metadata* section.
        """
        grp = h5file[f"sim_{sim_id}"]
        # Pull metadata – every concrete bundle stores at least a ``metadata`` section
        if "metadata" not in grp:
            raise KeyError(f"Bundle {sim_id} does not contain a 'metadata' section")
        meta = MetadataSection.from_group(grp["metadata"])
        bundle = cls._factory_from_metadata(meta)  # <-- implemented by subclass

        # Re‑attach the rest of the sections
        for name, subgrp in grp.items():
            if name in {"metadata", "raw_bundle"}:
                continue
            if name not in bundle.sections:
                raise KeyError(f"Unexpected section '{name}' in file")
            bundle.sections[name].from_hdf5(subgrp)

        # (optional) unpack raw bundle to a temporary directory
        if "raw_bundle" in grp:
            raw_blob = DataSection._read_binary_blob(grp["raw_bundle"])
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"sim_{sim_id}_"))
            with tempfile.TemporaryDirectory() as td:
                zip_path = Path(td) / "bundle.zip"
                zip_path.write_bytes(raw_blob)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp_dir)
            bundle._raw_extracted_path = tmp_dir  # just for convenience
        return bundle

    # Private helpers
    def _make_raw_bundle_blob(self) -> bytes:
        """
        Zip the whole ``original_root`` directory (preserving the internal layout)
        and return the bytes.  Uses the *deflate* algorithm (the same as a normal
        ``zip`` file) which yields good compression for text / json / csv and
        is fast for binary NumPy ``.npz`` files.
        """
        with tempfile.TemporaryDirectory() as td:
            zip_path = Path(td) / f"{self.sim_id}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in self.original_root.rglob("*"):
                    if f.is_file():
                        # store the path *relative* to the original root
                        rel = f.relative_to(self.original_root)
                        zf.write(f, arcname=str(rel))
            return zip_path.read_bytes()

    # Sub‑class hook – create a bundle from the (already parsed) metadata
    @classmethod
    @abc.abstractmethod
    def _factory_from_metadata(cls, meta: "MetadataSection") -> "SimulationBundle":
        """Instantiate the concrete bundle class *without* any sections populated."""
        ...
