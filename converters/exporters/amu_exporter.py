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
from ..schema.simulation_parameters import SimulationParameters
from ..schema.spike_map import SpikeMap
from ..schema.stimulus import Stimulus
from ..schema.stimulus_set import StimulusSet
from ..utils.helpers import normalize, AMU_POPULATIONS, get_discretized_stimulus, get_firing_rates

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
    output_suffix = ".amu.npz"

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
        sim_groups = [g for n, g in self.master["StandardFormat"]["Simulations"].items() if n.startswith("Sim_")]
        if not sim_groups:
            self.log.error("No simulation groups (sim_*) found in the master HDF5.")
            return

        self.log.info("Exporting %d simulations to %s (protocol=%s)",
                      len(sim_groups), self.out_dir, self.protocol_name)
        r = []
        u = []
        theta = []
        for sim_grp in sim_groups:
            sim_id = sim_grp.name.split("/", 1)[-1].replace("Sim_", "")
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
        The *sim_group* is now a ``Sim_<id>`` group that follows the new schema.
        """
        # ---- parameters -------------------------------------------------
        param_section = SimulationParameters.from_hdf5(sim_group["Parameters"])
        param_vals = np.array(
            [float(param_section.kv_pairs.get(p, np.nan)) for p in PARAMETERS],
            dtype=float,
        )

        # ---- stimulus ---------------------------------------------------
        stimulus_set = StimulusSet.from_hdf5(sim_group["Stimulus"])
        disc_stim = get_discretized_stimulus(stimulus_set, self.bin_size)

        # ---- spikes ------------------------------------------------------
        spike_map = SpikeMap.from_hdf5(sim_group["OutputSpike"])
        # Build the ``pop → {times, indices}`` dict expected by the utils.
        spikes: dict = {}
        neurons_count: dict = param_section.kv_pairs.get("record_neurons", {})
        for pop in spike_map.populations:
            # Concatenate all neuron spike‑times for this population.
            all_times = np.concatenate([n.times for n in pop.neurons]) if pop.neurons else np.empty(0)
            # AMU expects *indices* relative to the population – we create a dummy array.
            # (Indices are not used later in the calculation of get_firing_rates, only the times matter.)
            spikes[pop.population_name] = {"times": all_times, "indices": None}

        # ---- compute firing‑rate -----------------------------------------
        t0 = 0.0
        t_sim_ms = float(param_section.kv_pairs.get("t_sim", 1000.0))
        t1 = t_sim_ms / 1000.0

        _, firing_rate = get_firing_rates(
            spikes=spikes,
            dt=self.bin_size,
            t0=t0,
            t1=t1,
            nneurons=neurons_count,
        )

        self.log.info(
            "Simulation %s: firing_rate shape=%s, stimulus shape=%s, param_values=%s",
            sim_id,
            firing_rate.shape,
            disc_stim.shape,
            param_vals,
        )
        return {
            "firing_rate": firing_rate,
            "stimulus": disc_stim,
            "param_values": param_vals,
        }
