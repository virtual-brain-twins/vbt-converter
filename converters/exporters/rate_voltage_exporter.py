# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0

"""rate_voltage_exporter.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Utility for converting simulation data stored in a *Master* HDF5 file into
compressed NumPy ``.npz`` archives containing firing‑rate and membrane‑voltage
traces.  The :class:`RateVoltageExporter` class walks through all simulation
groups, extracts per‑population spike times and voltage recordings, optionally
smooths the resulting rate vectors and finally writes a dictionary of NumPy
arrays to disk.

Typical usage::

    parser = argparse.ArgumentParser()
    RateVoltageExporter.configure_parser(parser)
    args = parser.parse_args()

    with h5py.File(args.master_file, "r") as master_h5:
        exporter = RateVoltageExporter(
            master_h5,
            Path(args.out_dir),
            num_time_bins=args.num_time_bins,
            smoothing_win=args.smoothing_window,
            out_name=args.output_name,
        )
        exporter.run()
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np

from .protocol_exporter import ProtocolExporter
from ..schema.map import Map
from ..schema.population import Population
from ..schema.simulation import Simulation


class RateVoltageExporter(ProtocolExporter):
    """Export spike‑derived firing rates and voltage traces from a Master HDF5.

    The exporter iterates over all simulation groups (``Sim_*``) stored under
    ``/StandardFormat/Simulations``.  For each population it builds a binary spike
    matrix, optionally smooths the derived firing‑rate vector and averages any
    voltage traces that are present.  Global averages over all populations are
    also computed.  Results are saved as a compressed ``.npz`` file where the
    top‑level entry ``results`` contains a nested dictionary mapping population
    names to ``firing_rate`` and ``avg_voltage`` arrays.

    Attributes:
        num_time_bins: Number of time bins (1 ms resolution) used for rate
            computation.
        smoothing_win: Length of the moving‑average window applied to the raw
            firing‑rate vector.  A value of ``0`` disables smoothing.
        out_name: Filename of the output ``.npz`` archive (written inside
            :attr:`out_dir`).
    """

    @classmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        """Add command‑line arguments required by the exporter.

        Args:
            parser: An :class:`argparse.ArgumentParser` instance to which the
                arguments will be added.

        Raises:
            argparse.ArgumentError: If an argument with the same option string
                already exists in *parser*.
        """
        parser.add_argument(
            "-b",
            "--num-time-bins",
            type=int,
            default=10000,
            help="Number of time‑bins for the spike‑derived rate arrays (default: 10000).",
        )
        parser.add_argument(
            "-w",
            "--smoothing-window",
            type=int,
            default=10,
            help="Length of the smoothing window (in bins). Use 0 to disable smoothing.",
        )
        parser.add_argument(
            "-o",
            "--output-name",
            type=str,
            default="rates_voltages.npz",
            help="Name of the output .npz file (written to <out_dir>).",
        )

    def __init__(
        self,
        master_h5: h5py.File,
        out_dir: Path,
        num_time_bins: int = 10000,
        smoothing_win: int = 10,
        out_name: str = "rates_voltages.npz",
        log_level: int = logging.INFO,
    ) -> None:
        """Create a new exporter instance.

        Args:
            master_h5: Open HDF5 file handle pointing to the *Master* file that
                contains simulation results.
            out_dir: Directory where the resulting ``.npz`` file will be written.
            num_time_bins: Number of 1 ms time bins for the rate vectors.
            smoothing_win: Length of the moving‑average window applied to the raw
                rate (set to ``0`` to disable smoothing).
            out_name: Name of the output archive (e.g. ``rates_voltages.npz``).
            log_level: Logging verbosity for the exporter.

        Raises:
            TypeError: If *master_h5* is not an ``h5py.File`` or *out_dir* is not a
                :class:`pathlib.Path`.
        """
        super().__init__(master_h5, out_dir, log_level=log_level)
        self.num_time_bins = int(num_time_bins)
        self.smoothing_win = int(smoothing_win)
        self.out_name = out_name

    def run(self) -> None:
        """Process every simulation group and write the aggregated results.

        The method extracts data from all ``Sim_*`` groups, builds per‑population
        dictionaries, computes global averages and finally stores the complete
        mapping to a compressed ``.npz`` file.

        Raises:
            OSError: If the output directory cannot be created or the file cannot
                be written.
        """
        sim_root = self.master["StandardFormat"]["Simulations"]
        sim_groups = [g for n, g in sim_root.items() if n.startswith("Sim_")]

        if not sim_groups:
            self.log.error("No simulation groups (Sim_*) found in the master HDF5.")
            return

        self.log.info(
            "Exporting %d simulations to %s (output file = %s)",
            len(sim_groups),
            self.out_dir,
            self.out_name,
        )

        # Accumulate results in a plain dict – this will be pickled into the npz.
        all_results: Dict[str, Dict[str, Any]] = {}

        for sim_grp in sim_groups:
            # ``Sim_<ID>`` → extract just the numeric identifier.
            sim_id = sim_grp.name.split("/")[-1].replace("Sim_", "")
            try:
                result = self.extract_one_simulation(sim_id, sim_grp)
                all_results[sim_id] = result
            except Exception as exc:  # pragma: no cover – defensive programming
                self.log.exception("Failed to export simulation %s: %s", sim_id, exc)

        # Ensure the output directory exists.
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.out_dir / self.out_name

        # ``allow_pickle=True`` is required because we store a nested dict.
        np.savez_compressed(out_path, results=all_results, allow_pickle=True)

        self.log.info("Saved accumulated rate/voltage data to %s", out_path)

    def extract_one_simulation(
        self,
        sim_id: str,
        sim_group: h5py.Group,
    ) -> dict[str, Any]:
        """Extract firing‑rate and voltage information for a single simulation.

        The method loads the high‑level :class:`Simulation` object from *sim_group*,
        iterates over each population defined in the output map, builds spike
        matrices, computes (optionally smoothed) firing‑rate vectors and averages
        any voltage traces that are present.  Global averages across all
        populations are added under the ``"global_averages"`` key.

        Args:
            sim_id: Identifier of the simulation (numeric part extracted from the
                HDF5 group name).
            sim_group: HDF5 group containing the simulation data.

        Returns:
            A dictionary whose top‑level keys are population names and
            ``"global_averages"``, each mapping to another dictionary with
            ``"firing_rate"`` (``np.ndarray``) and ``"avg_voltage"`` (``np.ndarray``).

        Raises:
            Exception: Propagates any exception raised while constructing the
                :class:`Simulation` object (e.g. missing required groups or
                malformed data).
        """
        # Load the high‑level objects – this gives us a convenient Python
        # representation of the whole simulation.
        try:
            # The schema classes raise clear exceptions if required groups are
            # missing; we simply propagate them to the caller.
            sim_obj: Simulation = Simulation.from_hdf5(sim_group)
            # ``sim_obj.output_map`` is an instance of ``Map`` containing all
            # populations → neurons.
            output_map: Map = sim_obj.output_map
        except Exception as exc:
            self.log.error("Failed to load simulation %s: %s", sim_id, exc)
            raise

        # Prepare containers that will hold per‑population results.
        results: Dict[str, Any] = {}
        all_pop_firing_rates_for_global_avg: List[np.ndarray] = []
        all_pop_avg_voltages_for_global_avg: List[np.ndarray] = []

        # Iterate over every population present in the output map.
        for population in output_map.populations:  # type: Population
            pop_name = population.population_name

            # Gather meta‑information about the population
            total_neurons = len(population.neurons)

            # Build lookup tables that map a global neuron index (within this
            # population) → row index in the *full* matrix and → row index in the
            # *recorded‑only* matrix.
            g2l_total = {n.index: i for i, n in enumerate(population.neurons)}

            #   Allocate the two spike matrices (rate‑calc matrix always needed)
            spike_matrix_for_rate_calc = (
                np.zeros((total_neurons, self.num_time_bins), dtype=int)
                if total_neurons > 0
                else np.empty((0, 0), dtype=int)
            )

            # Voltage handling – we must collect traces of equal length.
            voltage_traces: List[np.ndarray] = []
            expected_voltage_len: int | None = None
            any_spikes_processed = False

            # Process recorded neurons only (virtual neurons have no data)
            for neuron in population.neurons:
                # spikes
                if neuron.times.size > 0:
                    any_spikes_processed = True
                    # Convert seconds → integer bin index (1 ms resolution)
                    col_idx = np.rint(neuron.times * 1000).astype(int)
                    # Clip to the allowed range
                    valid = (col_idx >= 0) & (col_idx < self.num_time_bins)
                    col_idx = col_idx[valid]
                    if col_idx.size > 0:
                        # Populate the full matrix (used for rate calculation)
                        row_total = g2l_total[neuron.index]
                        spike_matrix_for_rate_calc[row_total, col_idx] = 1

                # voltage
                # ``neuron.voltage`` may be empty
                if neuron.voltage.size > 0:
                    vt = neuron.voltage
                    if expected_voltage_len is None:
                        expected_voltage_len = vt.size
                        voltage_traces.append(vt)
                    elif vt.size == expected_voltage_len:
                        voltage_traces.append(vt)
                    else:
                        self.log.warning(
                            "Voltage trace length mismatch for neuron %d (%s): "
                            "got %d, expected %d – skipping this trace",
                            neuron.index,
                            pop_name,
                            vt.size,
                            expected_voltage_len,
                        )
                # If the voltage array is empty we simply ignore it – the original
                # helper behaved the same way.

            # Compute the *population* firing‑rate
            if total_neurons > 0 and spike_matrix_for_rate_calc.size > 0:
                raw_rate = np.mean(spike_matrix_for_rate_calc, axis=0)  # shape (num_bins,)
            else:
                raw_rate = np.zeros(self.num_time_bins)

            if self.smoothing_win > 0 and total_neurons > 0:
                window = np.ones(self.smoothing_win, dtype=float) / self.smoothing_win
                smoothed_rate = np.convolve(raw_rate, window, mode="same")
            else:
                smoothed_rate = raw_rate

            # Store per‑population results
            results[pop_name] = {
                "firing_rate": smoothed_rate,
                "avg_voltage": np.array([]),  # will be filled later if possible
            }

            # Store the firing‑rate for the global average
            all_pop_firing_rates_for_global_avg.append(smoothed_rate)

            # Average voltage traces (if any) and store them
            if voltage_traces:
                avg_voltage = np.mean(np.stack(voltage_traces), axis=0)
                results[pop_name]["avg_voltage"] = avg_voltage
                all_pop_avg_voltages_for_global_avg.append(avg_voltage)
            else:
                # No valid voltage traces – keep an empty array (mirrors legacy)
                results[pop_name]["avg_voltage"] = np.array([])

            # Informational messages (exactly the same wording as the legacy
            #     helper – useful for debugging and for unit‑tests that rely on
            #     stdout/stderr).
            if not any_spikes_processed and total_neurons > 0:
                self.log.info(
                    "Info: No spikes found/processed for any recorded neuron in population '%s' (simulation %s).",
                    pop_name,
                    sim_id,
                )
            elif total_neurons == 0:
                self.log.info(
                    "Info: Population '%s' has zero total neurons according to HDF5 metadata (simulation %s).",
                    pop_name,
                    sim_id,
                )

        # Compute *global* averages
        global_avg: Dict[str, np.ndarray] = {}

        if all_pop_firing_rates_for_global_avg:
            global_avg["firing_rate"] = np.mean(
                np.stack(all_pop_firing_rates_for_global_avg), axis=0
            )
        else:
            global_avg["firing_rate"] = np.zeros(self.num_time_bins)

        if all_pop_avg_voltages_for_global_avg:
            # Ensure all voltage traces have the same length before averaging.
            first_len = -1
            for v in all_pop_avg_voltages_for_global_avg:
                if v.size > 0:
                    first_len = v.shape[0]
                    break
            if first_len != -1 and all(
                v.shape[0] == first_len for v in all_pop_avg_voltages_for_global_avg if v.size > 0
            ):
                valid_traces = [v for v in all_pop_avg_voltages_for_global_avg if v.size > 0]
                if valid_traces:
                    global_avg["avg_voltage"] = np.mean(np.stack(valid_traces), axis=0)
                else:
                    global_avg["avg_voltage"] = np.array([])
            else:
                if first_len == -1:
                    global_avg["avg_voltage"] = np.array([])
                else:
                    self.log.warning(
                        "Warning: Population average voltage traces have inconsistent lengths – "
                        "cannot compute a meaningful global average voltage."
                    )
                    global_avg["avg_voltage"] = np.array([])
        else:
            global_avg["avg_voltage"] = np.array([])

        results["global_averages"] = global_avg

        return results

    #  Compatibility shim required by the abstract base class.
    def export_one_simulation(self, sim_id: str, sim_group: h5py.Group) -> None:
        pass
