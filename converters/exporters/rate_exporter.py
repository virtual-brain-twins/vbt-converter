"""rate_exporter.py – Export rate data from a master HDF5 file.

The :class:`RateExporter` class iterates over all simulation groups in a
``StandardFormat/Simulations`` hierarchy, extracts the stimulus,
spike‑times and simulation parameters, computes population firing‑rates,
normalises the data, and finally stores everything in a compressed
``.npz`` file that can be consumed by downstream analysis pipelines.

Typical usage (via the generic driver) looks like::

    from pathlib import Path
    import h5py
    from rate_exporter import RateExporter

    master = h5py.File("master.h5", "r")
    out_dir = Path("./exported")
    exporter = RateExporter(master, out_dir, bin_size=0.001)
    exporter.run()
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy import dtype, ndarray

from .protocol_exporter import ProtocolExporter
from ..schema.simulation import Simulation
from ..utils.helpers import normalize, get_discretized_stimulus, get_firing_rates


class RateExporter(ProtocolExporter):
    """Export a collection of simulations in the *UVA* protocol format.

    The exporter follows the generic :class:`~.protocol_exporter.ProtocolExporter`
    contract: the constructor receives a master HDF5 file, an output directory,
    and any protocol‑specific arguments (here only the time‑bin size).  The
    :meth:`run` method walks through all simulation groups, delegates the heavy
    lifting to :meth:`extract_one_simulation`, normalises the results and writes
    a ``.npz`` file that contains:

    * ``x`` – normalised firing‑rates (shape: ``(npoints, nt, nvar)``)
    * ``u`` – normalised stimulus (shape: ``(npoints, nt, 1)``)
    * ``theta`` – normalised parameter vector (shape: ``(npoints, nparams)``)
    * auxiliary meta‑information (population names, parameter names, scaling
      factors, and the time‑step ``dt``).

    Attributes
    ----------
    master_h5 : h5py.File
        The opened master HDF5 file that stores all simulations.
    out_dir : pathlib.Path
        Destination directory for the exported ``.npz`` file.
    bin_size : float
        Size of the time bin (seconds) used to discretise spikes and stimulus.
    log : logging.Logger
        Logger used throughout the exporter (created by the base class).
    """

    # Public protocol information (used by the generic driver)
    protocol_name = "uva_amu_1"
    """Name of the protocol – used to form output file names."""

    output_suffix = ".npz"
    """File‑extension for the exported data."""

    # Argument handling
    @classmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        """Add protocol‑specific command‑line arguments to *parser*.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The parser that will be extended with a ``--bin-size`` option.
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
        """Create an :class:`RateExporter`.

        Parameters
        ----------
        master_h5 : h5py.File
            Open HDF5 file that follows the *StandardFormat* schema.
        out_dir : pathlib.Path
            Directory where the ``.npz`` output will be written.
        bin_size : float
            Desired discretisation interval (seconds) for both spikes and stimulus.
        log_level : int, optional
            Logging level passed to the base class (default: ``logging.INFO``).
        """
        super().__init__(master_h5, out_dir, log_level=log_level)
        self.bin_size = bin_size

    # Main driver
    def run(self) -> None:
        """Iterate over all simulation groups and export them.

        The method collects the raw results of :meth:`extract_one_simulation`
        for every simulation, normalises the concatenated arrays and stores the
        final compressed ``.npz`` file in ``self.out_dir``.  Any simulation that
        raises an exception is skipped, with a full stack‑trace logged at
        ``ERROR`` level.
        """
        # Locate all simulation groups (named ``Sim_<ID>``) under the standard tree.
        sim_groups = [
            g
            for n, g in self.master["StandardFormat"]["Simulations"].items()
            if n.startswith("Sim_")
        ]
        if not sim_groups:
            self.log.error("No simulation groups (sim_*) found in the master HDF5.")
            return

        self.log.info(
            "Exporting %d simulations to %s (protocol=%s)",
            len(sim_groups),
            self.out_dir,
            self.protocol_name,
        )
        r, u, theta = [], [], []
        unique_populations = set()
        result = {}
        for sim_grp in sim_groups:
            # The HDF5 name is ``/StandardFormat/Simulations/Sim_<ID>`` – we keep only the <ID>.
            sim_id = sim_grp.name.split("/", 1)[-1].replace("Sim_", "")
            try:
                result = self.extract_one_simulation(sim_id, sim_grp)

                # ``firing_rate`` has shape (nt, nvar); we store the transpose to get (nvar, nt).
                r.append(result["firing_rate"].T)
                # ``stimulus`` is a 1‑D array – we expand the last dimension to match the model’s expected shape.
                u.append(result["stimulus"][:, None])
                theta.append(result["param_values"])
                unique_populations.update(result["populations"])
            except Exception as exc:  # pragma: no cover – defensive programming
                self.log.exception("Failed to export simulation %s: %s", sim_id, exc)

        # Convert collected Python lists into NumPy arrays.
        r = np.array(r)  # shape: (npoints, nt, nvar)
        u = np.array(u)  # shape: (npoints, nt, 1)
        theta = np.array(theta)  # shape: (npoints, nparams)

        # Normalisation
        r_norm, mu_r, sd_r = normalize(r, axis=(0, 1))
        u_norm, mu_u, sd_u = normalize(u, axis=(0, 1))
        theta_norm, mu_theta, sd_theta = normalize(theta, axis=0)

        # Persist the data set
        out_path = self.out_dir / f"{self.protocol_name}{self.output_suffix}"
        np.savez_compressed(
            out_path,
            x=r_norm,
            u=u_norm,
            theta=theta_norm,
            names_x=list(unique_populations),
            names_theta=result["params"],
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
    def extract_one_simulation(
            self,
            sim_id: str,
            sim_group: h5py.Group,
    ) -> dict[
        str,
        ndarray[tuple[Any, ...], dtype[Any]] | list[str],
    ]:
        """Extract the raw data for a single simulation.

        The helper reads three distinct sections from the HDF5 file,
        builds a unified spike‑dictionary, computes binned firing‑rates and
        returns everything required by :meth:`run`.

        Parameters
        ----------
        sim_id : str
            Human‑readable identifier of the simulation (derived from the group name).
        sim_group : h5py.Group
            HDF5 group that contains the ``Parameters``, ``Configuration``,
            ``Stimuli`` and ``Output`` sub‑groups for a single simulation.

        Returns
        -------
        dict
            Mapping with the following keys:

            * ``"firing_rate"`` – ``np.ndarray`` of shape ``(nt, nvar)`` containing
              the population firing‑rates binned at ``self.bin_size``.
            * ``"stimulus"`` – 1‑D ``np.ndarray`` holding the discretised stimulus.
            * ``"param_values"`` – 1‑D ``np.ndarray`` of the three protocol parameters
              (order follows :data:`PARAMETERS`).
            * ``"populations"`` – list of population names present in the spike map.
        """

        try:
            # The schema classes raise clear exceptions if required groups are
            # missing; we simply propagate them to the caller.
            sim_obj: Simulation = Simulation.from_hdf5(sim_group)
        except Exception as exc:
            self.log.error("Failed to load simulation %s: %s", sim_id, exc)
            raise

        # Pull out the parameters of interest in the order defined by PARAMETERS.
        param_vals = np.fromiter(
            (float(sim_obj.parameters.get_value(p)) for p in sim_obj.parameters.iter_keys()),
            dtype=float,
        )

        # Discretise the stimulus
        disc_stim = get_discretized_stimulus(sim_obj.stimuli, self.bin_size)

        # ``spikes`` maps ``population_name → {"times": np.ndarray, "indices": …}``.
        spikes: dict[str, dict[str, Any]] = {}
        # Number of recorded neurons per population (required for firing‑rate scaling).
        neurons_count: dict[str, int] = sim_obj.configuration.get_value("record_neurons")

        # Keep a list of population names to forward back to the caller.
        populations = [pop.population_name for pop in sim_obj.output_map.populations]

        for pop in sim_obj.output_map.populations:
            # Concatenate spike times of *all* neurons that belong to this population.
            all_times = np.concatenate([n.times for n in pop.neurons]) if pop.neurons else np.empty(0)

            # The UVA protocol does not need per‑neuron indices – they are
            # kept as ``None`` to simplify downstream processing.
            spikes[pop.population_name] = {"times": all_times, "indices": None}

        # Compute firing‑rates
        t0 = 0.0  # start of the simulation (seconds)
        t_sim_ms = float(sim_obj.configuration.get_value("t_sim"))  # total simulation time (ms)
        t1 = t_sim_ms / 1000.0  # convert to seconds

        # ``get_firing_rates`` returns a tuple ``(times, rates)`` – we keep only the rates.
        _, firing_rate = get_firing_rates(
            spikes=spikes,
            dt=self.bin_size,
            t0=t0,
            t1=t1,
            nneurons=neurons_count,
            populations=populations,
        )

        self.log.info(
            "Simulation %s: firing_rate shape=%s, stimulus shape=%s, param_values=%s, params=%s",
            sim_id,
            firing_rate.shape,
            disc_stim.shape,
            param_vals,
            sim_obj.parameters.keys(),
        )
        return {
            "firing_rate": firing_rate,
            "stimulus": disc_stim,
            "params": sim_obj.parameters.keys(),
            "param_values": param_vals,
            "populations": populations,
        }

    # Placeholder for the abstract method required by the base class.
    def export_one_simulation(self, sim_id: str, sim_group: h5py.Group) -> None:
        """Compatibility shim – the concrete logic lives in :meth:`extract_one_simulation`.

        The generic :class:`ProtocolExporter` expects a method named
            ``export_one_simulation``.  In the UVA implementation the heavy lifting
        is performed by :meth:`extract_one_simulation`; this stub simply
        satisfies the abstract‐base‑class contract and is never called
        directly by the driver.
        """
        pass
