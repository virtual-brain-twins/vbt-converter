# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0
#
import abc
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from converters.utils.common import configure_logging


class ProtocolExporter(abc.ABC):
    """
    Base class for all “master‑HDF5 → protocol‑specific” exporters.

    A concrete subclass must implement :meth:`export_one_simulation` – the
    routine that receives the *simulation group* (the `sim_<id>` group) and
    writes the output for that simulation.
    """

    # Human-readable name used for the CLI help text
    protocol_name: str = "base"

    # Extension that will be appended to each output file when the exporter
    # creates one per simulation (e.g. `.npz`, `.dsr.npz`, …) or for the whole list of simulations.
    output_suffix: str = ".out"

    # If the exporter needs extra CLI options (e.g. `--bin-size` for UVA)
    # they can be added by the class method ``configure_parser``.
    @classmethod
    def configure_parser(cls, parser: "argparse.ArgumentParser") -> None:  # pragma: no cover
        """Add protocol‑specific arguments to the parser."""
        pass

    def __init__(self, master_h5: h5py.File, out_dir: Path, log_level: int = logging.INFO):
        """
        Parameters
        ----------
        master_h5 : h5py.File
            Already opened master file (read‑only).
        out_dir : pathlib.Path
            Directory in which the exporter should write its results.
        log_level : int, optional
            Verbosity for the exporter’s own logger.
        """
        self.master = master_h5
        self.out_dir = out_dir
        self.log = configure_logging(
            module_name=self.__class__.__module__ + "." + self.__class__.__name__,
            level=log_level,
            log_file_name=Path(__file__).with_suffix(".log").name,
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # Helper utilities that concrete exporters will likely need
    @staticmethod
    def _as_numpy(dataset: h5py.Dataset) -> np.ndarray:
        """Return a *real* numpy array – reading lazily if the dataset is chunked."""
        return np.asarray(dataset)

    @staticmethod
    def _read_section(group: h5py.Group, name: str) -> h5py.Group:
        """Convenient wrapper that raises a nice error if a section is missing."""
        if name not in group:
            raise KeyError(f"Section '{name}' not found in simulation group '{group.name}'.")
        return group[name]

    # Public API – called by the CLI driver
    def run(self) -> None:
        """Iterate over all `sim_<id>` groups and hand each to the concrete exporter."""
        sim_groups = [g for n, g in self.master.items() if n.startswith("sim_")]
        if not sim_groups:
            self.log.error("No simulation groups (sim_*) found in the master HDF5.")
            return

        self.log.info("Exporting %d simulations to %s (protocol=%s)",
                      len(sim_groups), self.out_dir, self.protocol_name)

        for sim_grp in sim_groups:
            sim_id = sim_grp.name.split("/", 1)[-1].replace("sim_", "")
            try:
                self.export_one_simulation(sim_id, sim_grp)
            except Exception as exc:  # pragma: no cover – defensive
                self.log.exception("Failed to export simulation %s: %s", sim_id, exc)

    @abc.abstractmethod
    def export_one_simulation(self, sim_id: str, sim_group: h5py.Group) -> None:
        """
        Perform the actual conversion for a *single* simulation.

        The method must write its result into ``self.out_dir`` – the filename
        should be ``{sim_id}{self.output_suffix}``.
        """
        ...

    @abc.abstractmethod
    def extract_one_simulation(self, sim_id: str, sim_group: h5py.Group) -> Any:
        """
        Perform the actual conversion for a *single* simulation to any needed format and returns the result.
        """
        ...
