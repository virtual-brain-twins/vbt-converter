"""
AMU exporter

This module implements :class:`AMUExporter`, the concrete exporter that
converts a *master* HDF5 file (produced by :class:`~converters.bundles.amu_bundle.AMUBundle`)
into the *DS‑R* ``.npz`` format required by the AMU analysis pipeline.

*   scalar simulation parameters are taken from the **attributes** of each
    ``sim_<id>`` group.
*   spike‑times are stored under ``sim_<id>/spike_trains/<population>/times``.
*   the number of recorded neurons is stored as a *binary* payload in the
    dataset ``sim_<id>/record_neurons`` – it is loaded with ``np.load``.
*   the external stimulus is stored in the group ``sim_<id>/stimulus`` and is
    piece‑wise‑constant.
*   After the raw firing‑rates, stimulus and parameters are **normalised**
    (zero‑mean, unit‑variance) the arrays are saved with ``np.savez`` using the
    naming convention expected by the downstream code.

The class inherits from :class:`~converters.interfaces.protocol_exporter.ProtocolExporter`
so that it plugs‑in to the generic command‑line driver used by the package.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from .protocol_exporter import ProtocolExporter
from .utils import (
    AMU_POPULATIONS,  # list of population names used by the pipeline
    get_discretized_stimulus,
    get_firing_rates,
    normalize,
)
from ..sections.amu.record_neurons_section import RecordNeuronsSection
from ..sections.amu.spiketrain_section import SpikeTrainSection
from ..sections.amu.stimulus_section import StimulusSection
from ..sections.metadata_section import MetadataSection

# Fixed definitions – they must stay identical to the reference script
PARAMETERS = ["scale_num_neurons", "scale_syn_strength", "scale_nmda_efficacy"]
PARAMETERS_SHORT = ["numneur", "synstr", "nmdaeff"]


class AMUExporter(ProtocolExporter):
    """
    Export a master HDF5 file (generated from many :class:`AMUBundle`s) into the
    DS‑R ``.npz`` format used by the AMU pipeline.

    The exporter works on a *per‑simulation* basis – all simulation groups are iterated and the final results are calculated.
    """

    def export_one_simulation(self, sim_id: str, sim_group: h5py.Group) -> None:
        pass

    # Public protocol information (used by the generic driver)
    protocol_name = "amu"
    output_suffix = ".dsr.npz"

    # Argument handling
    @classmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        """
        Add the ``--bin-size`` argument required by the AMU conversion.
        """
        parser.add_argument(
            "-b",
            "--bin-size",
            type=float,
            required=True,
            help="Bin size (seconds) used to discretise spikes and stimulus.",
        )

    # Construction
    def __init__(
            self,
            master_h5: h5py.File,
            out_dir: Path,
            bin_size: float,
            log_level: int = logging.INFO,
    ) -> None:
        """
        Parameters
        ----------
        master_h5
            Open ``h5py.File`` object that contains the *simulations* group.
        out_dir
            Directory where the ``*.dsr.npz`` files will be written.
        bin_size
            Size of the time‑bin (in seconds) for firing‑rate discretisation.
        log_level
            Logging verbosity.
        """
        super().__init__(master_h5, out_dir, log_level=log_level)
        self.bin_size = bin_size

    def run(self) -> None:
        """Iterate over all `sim_<id>` groups and hand each to the concrete exporter."""
        sim_groups = [g for n, g in self.master.items() if n.startswith("sim_")]
        if not sim_groups:
            self.log.error("No simulation groups (sim_*) found in the master HDF5.")
            return

        self.log.info("Exporting %d simulations to %s (protocol=%s)",
                      len(sim_groups), self.out_dir, self.protocol_name)
        r = []
        u = []
        theta = []
        for sim_grp in sim_groups:
            sim_id = sim_grp.name.split("/", 1)[-1].replace("sim_", "")
            try:
                result = self.extract_one_simulation(sim_id, sim_grp)
                r.append(result["firing_rate"].T)
                u.append(result["stimulus"][:, None])
                theta.append(result["param_values"])
            except Exception as exc:  # pragma: no cover – defensive
                self.log.exception("Failed to export simulation %s: %s", sim_id, exc)
        r = np.array(r)  # shape: (npoints, nt, nvar)
        u = np.array(u)  # shape: (npoints, nt, 1)
        theta = np.array(theta)  # shape: (npoints, nparams)

        # Normalize
        r_norm, mu_r, sd_r = normalize(r, axis=(0, 1))
        u_norm, mu_u, sd_u = normalize(u, axis=(0, 1))
        theta_norm, mu_theta, sd_theta = normalize(theta, axis=0)
        out_path = self.out_dir / f"{self.protocol_name}{self.output_suffix}"
        np.savez_compressed(
            out_path,
            x=r_norm,
            u=u_norm,
            theta=theta_norm,
            names_x=AMU_POPULATIONS,
            names_theta=PARAMETERS_SHORT,
            dt=self.bin_size,
            mu_x=mu_r,
            sd_x=sd_r,
            mu_u=mu_u,
            sd_u=sd_u,
            mu_theta=mu_theta,
            sd_theta=sd_theta,
        )
        self.log.info("Saved DS-R .npz file for simulation: %s", out_path)

    # Core conversion per simulation
    def extract_one_simulation(self, sim_id: str, sim_group: h5py.Group) -> dict[str, np.ndarray]:
        """
        Convert a single simulation group into a .

        Expected layout (produced by :class:`AMUBundle`):

        ``sim_<id>/``
            ├─ attributes                – scalar parameters (e.g. ``scale_num_neurons``)
            ├─ ``spike_trains``          – per‑population spike times
            ├─ ``record_neurons``        – binary payload (structured array)
            ├─ ``stimulus``              – piece‑wise‑constant input
            └─ (other sections that are ignored by the exporter)
        """

        # Read the metadata section of the simulation and extract the scalar parameters
        metadata_section = MetadataSection.from_group(sim_group["metadata"])  # validate metadata section
        param_vals = np.array(
            [float(metadata_section.data.get(p, np.nan)) for p in PARAMETERS],
            dtype=float,
        )

        # Load the spike‑train data
        spikes = SpikeTrainSection.from_group(sim_group["spike_trains"]).data

        # Load the neuron‑count table
        nneurons = RecordNeuronsSection.from_group(sim_group["record_neurons"]).data

        # Compute firing‑rates (Hz) for each population
        t0 = 0.0
        # ``t_sim`` is stored in *milliseconds* as an attribute of the simulation group.
        t_sim_ms = float(metadata_section.data.get("t_sim", 1000.0))
        t1 = t_sim_ms / 1000.0  # convert to seconds

        _, firing_rate = get_firing_rates(
            spikes,
            self.bin_size,
            t0=t0,
            t1=t1,
            nneurons=nneurons,
        )

        # Discretize the stimulus
        external_stimulus = StimulusSection.from_group(sim_group["stimulus"])
        stimulus = get_discretized_stimulus(external_stimulus, self.bin_size)
        self.log.info("Simulation %s: firing_rate shape=%s, stimulus shape=%s, param_values=%s",)
        return {
            "firing_rate": firing_rate,
            "stimulus": stimulus,
            "param_values": param_vals
        }
