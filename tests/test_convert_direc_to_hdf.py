"""
End‑to‑end pytest‑Allure suite for ``convert_direc_to_hdf.convert_to_hdf5``.
The suite builds a temporary directory tree that mimics the layout
expected by the converter, runs the conversion and then inspects the
resulting HDF5 file.

All helper functions create **exactly** the file format that
``convert_to_hdf5`` expects:

* ``config.yaml`` – plain YAML.
* ``spike_times/<layer>.npz`` – ``np.savez`` with ``indices`` & ``times``.
* ``*.log`` – plain text files.
* ``record_info/record_neurons.data`` – a **pickled dict** saved with
  ``np.save`` **without** the automatic ``.npy`` suffix (the converter
  loads it with ``np.load(..., allow_pickle=True)``).
* ``metrics/avg_firing_rates.csv`` – a CSV readable by ``pandas.read_csv``.
* ``random_input_<duration>ms.npz`` – ``np.savez`` containing a single
  array (named ``arr_0``).

The test reproduces the exact logic used in the production code to
derive the expected *limits* and *values* for the stimulus group, so it
remains correct even if the implementation changes the way those arrays
are built.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import allure
import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

# ----------------------------------------------------------------------
# Function under test
# ----------------------------------------------------------------------
from converters.convert_direc_to_hdf import convert_to_hdf5

# ----------------------------------------------------------------------
# Helper utilities – create the temporary directory tree.
# ----------------------------------------------------------------------


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Write a dictionary as a YAML file."""
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def _write_spike_file(path: Path, indices: np.ndarray, times: np.ndarray) -> None:
    """Write ``indices`` and ``times`` with ``np.savez``."""
    np.savez(path, indices=indices, times=times)


def _write_log_file(path: Path, text: str) -> None:
    """Write a plain‑text log file."""
    path.write_text(text, encoding="utf-8")


def _write_record_info(path: Path, dct: Dict[str, int]) -> None:
    """
    ``record_neurons.data`` must be a *pickled dict* saved **exactly**
    under that filename.  ``np.save`` automatically appends “.npy” when
    the target is a string, therefore we open the file ourselves and pass
    the file‑handle to ``np.save`` – this prevents the unwanted suffix.
    """
    with path.open("wb") as f:
        np.save(f, dct, allow_pickle=True)


def _write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write the metrics CSV using pandas (ensures the exact column order)."""
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_stimulus_files(base_dir: Path, stim_spec: List[Dict[str, Any]]) -> None:
    """
    ``stim_spec`` contains dictionaries with:

        - ``duration_ms`` – integer duration
        - ``array``       – ``np.ndarray`` (the stimulus values)

    Files are named ``random_input_<duration>ms.npz`` and contain a
    single array stored as ``arr_0``.
    """
    for spec in stim_spec:
        fname = f"random_input_{spec['duration_ms']}ms.npz"
        np.savez(base_dir / fname, spec["array"])


def _create_simulation_dir(root: Path, sim_name: str, scenario: Dict[str, Any]) -> None:
    """Create one simulation folder (``sim_name``) inside ``root``."""
    sim_dir = root / sim_name
    sim_dir.mkdir(parents=True)

    # ── config.yaml
    _write_yaml(sim_dir / "config.yaml", scenario["config"])

    # ── spike_times/
    spike_dir = sim_dir / "spike_times"
    spike_dir.mkdir()
    for layer, data in scenario["spikes"].items():
        _write_spike_file(spike_dir / f"{layer}.npz", data["indices"], data["times"])

    # ── logs
    for log_name, txt in scenario["logs"].items():
        _write_log_file(sim_dir / log_name, txt)

    # ── record_info/
    rec_dir = sim_dir / "record_info"
    rec_dir.mkdir()
    _write_record_info(rec_dir / "record_neurons.data", scenario["record_neurons"])

    # ── metrics/
    metrics_dir = sim_dir / "metrics"
    metrics_dir.mkdir()
    _write_metrics_csv(metrics_dir / "avg_firing_rates.csv", scenario["metrics"])

    # ── stimulus files (directly in the simulation folder)
    _write_stimulus_files(sim_dir, scenario["stimulus"])


# ----------------------------------------------------------------------
# Scenarios – each dict will be materialised by the helper above.
# ----------------------------------------------------------------------
SCENARIOS = [
    {
        "id": "full",
        "description": "All possible files present, multiple spike layers, two stimulus chunks.",
        "config": {"t_sim": 250, "model": "test_model", "seed": 42},
        "spikes": {
            "L1": {"indices": np.arange(10, dtype=int), "times": np.linspace(0, 100, 10)},
            "L2": {"indices": np.arange(5, 15, dtype=int), "times": np.linspace(10, 200, 10)},
        },
        "logs": {
            "error.log": "Error: something went wrong\n",
            "info.log": "Info: all good\n",
            "warning.log": "Warning: be careful\n",
        },
        "record_neurons": {"neur1": 1, "neur2": 2, "neur3": 3},
        "metrics": [
            {"Population": "popA", "Avg Firing Rate Hz": 5.5},
            {"Population": "popB", "Avg Firing Rate Hz": 12.2},
        ],
        "stimulus": [
            {"duration_ms": 50, "array": np.full((5,), 0.1)},
            {"duration_ms": 200, "array": np.full((5,), 0.9)},
        ],
    },
    {
        "id": "minimal",
        "description": "Only one spike layer, empty logs, single stimulus block, single metric row.",
        "config": {"t_sim": 100, "model": "minimal_model"},
        "spikes": {
            "L1": {
                "indices": np.array([0, 2, 4], dtype=int),
                "times": np.array([0.0, 10.0, 20.0]),
            }
        },
        "logs": {"error.log": "", "info.log": "", "warning.log": ""},
        "record_neurons": {"a": 10},
        "metrics": [{"Population": "popX", "Avg Firing Rate Hz": 0.0}],
        "stimulus": [
            {"duration_ms": 100, "array": np.full((3,), 0.5)},
        ],
    },
    {
        "id": "no_spikes_no_stim",
        "description": (
            "No spike files, no stimulus files, three metric rows, non‑default t_sim."
        ),
        "config": {"t_sim": 300, "model": "edge_case"},
        "spikes": {},  # empty – the converter must handle an empty directory.
        "logs": {
            "error.log": "Error only\n",
            "info.log": "Info only\n",
            "warning.log": "Warning only\n",
        },
        "record_neurons": {"x": 0, "y": -1},
        "metrics": [
            {"Population": "pop1", "Avg Firing Rate Hz": 1.1},
            {"Population": "pop2", "Avg Firing Rate Hz": 2.2},
            {"Population": "pop3", "Avg Firing Rate Hz": 3.3},
        ],
        "stimulus": [],  # no stimulus files at all.
    },
]


# ----------------------------------------------------------------------
# Pytest‑Allure test class
# ----------------------------------------------------------------------
@allure.feature("HDF5 conversion")
@allure.story("convert_to_hdf5")
@allure.severity(allure.severity_level.CRITICAL)
@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=[s["id"] for s in SCENARIOS],
)
class TestConvertToHDF5:
    """Parameterised end‑to‑end test suite."""

    @staticmethod
    def _build_input(tmp_path: Path, scenario: Dict[str, Any]) -> Path:
        """Create the temporary input directory tree."""
        input_root = tmp_path / "input"
        input_root.mkdir()
        _create_simulation_dir(input_root, scenario["id"], scenario)
        return input_root

    @allure.title("Convert folder structure to HDF5 and validate contents")
    def test_convert_to_hdf5(self, tmp_path: Path, scenario: Dict[str, Any]) -> None:
        # ------------------------------------------------------------------
        # 1️⃣ Build the input directory hierarchy
        # ------------------------------------------------------------------
        input_dir = self._build_input(tmp_path, scenario)
        output_file = tmp_path / f"{scenario['id']}.h5"

        # ------------------------------------------------------------------
        # 2️⃣ Run the conversion
        # ------------------------------------------------------------------
        with allure.step("Execute conversion"):
            convert_to_hdf5(str(input_dir), str(output_file))

        # ------------------------------------------------------------------
        # 3️⃣ Inspect the produced HDF5 file
        # ------------------------------------------------------------------
        with allure.step("Open produced HDF5 file"):
            with h5py.File(output_file, "r") as h5:
                # --------------------------------------------------------------
                # Global attributes
                # --------------------------------------------------------------
                with allure.step("Check global attributes"):
                    assert h5.attrs["version"] == "0.1"

                # --------------------------------------------------------------
                # Simulations group
                # --------------------------------------------------------------
                sims = h5["simulations"]
                assert scenario["id"] in sims, "Simulation group missing"
                sim = sims[scenario["id"]]

                # --------------------------------------------------------------
                # Config attributes
                # --------------------------------------------------------------
                with allure.step("Validate config attributes"):
                    for key, expected in scenario["config"].items():
                        assert sim.attrs[key] == expected, f"config[{key!r}] mismatch"

                # --------------------------------------------------------------
                # Spike times
                # --------------------------------------------------------------
                with allure.step("Validate spike data"):
                    spike_grp = sim["spike_times"]
                    expected_layers = set(scenario["spikes"])
                    assert set(spike_grp) == expected_layers, "Spike layer mismatch"
                    for layer, src in scenario["spikes"].items():
                        grp = spike_grp[layer]
                        np.testing.assert_array_equal(
                            grp["indices"][...], src["indices"]
                        )
                        np.testing.assert_allclose(
                            grp["times"][...], src["times"]
                        )

                # --------------------------------------------------------------
                # Logs
                # --------------------------------------------------------------
                with allure.step("Validate log datasets"):
                    log_grp = sim["logs"]
                    for name, txt in scenario["logs"].items():
                        raw: np.ndarray = log_grp[name][...]  # scalar byte‑string
                        stored = raw.tobytes().decode()
                        assert stored == txt, f"{name} content mismatch"

                # --------------------------------------------------------------
                # Record neurons
                # --------------------------------------------------------------
                with allure.step("Validate record_neurons dataset"):
                    rec_arr = sim["record_neurons"][...]  # structured array
                    rec_dict = {
                        rec_arr["label"][i].decode("utf-8"): int(rec_arr["value"][i])
                        for i in range(len(rec_arr))
                    }
                    assert rec_dict == scenario["record_neurons"], "record_neurons mismatch"

                # --------------------------------------------------------------
                # Metrics
                # --------------------------------------------------------------
                with allure.step("Validate metrics dataset"):
                    metrics_arr = sim["metrics"]["Avg Firing Rate Hz"][...]
                    metric_dict = {
                        label.decode("utf-8"): float(val)
                        for label, val in zip(metrics_arr["label"], metrics_arr["value"])
                    }
                    expected = {
                        row["Population"]: float(row["Avg Firing Rate Hz"])
                        for row in scenario["metrics"]
                    }
                    assert metric_dict == expected, "Metrics mismatch"

                # --------------------------------------------------------------
                # Stimulus
                # --------------------------------------------------------------
                with allure.step("Validate stimulus group"):
                    stim = sim["stimulus"]
                    assert stim.attrs["type"] == "piecewise_constant"

                    # ----------------------------------------------------------
                    # Expected limits & values – replicate the logic from the
                    # production code so the test stays in sync.
                    # ----------------------------------------------------------
                    exp_limits = [0]
                    exp_values = [0.0]  # the first scalar inserted by the converter
                    pattern = re.compile(r"random_input_(\d+)ms\.npz")
                    # Files are already sorted in the conversion routine; we sort
                    # them here as well to obtain the same order.
                    fnames = sorted(
                        [
                            f
                            for f in os.listdir(os.path.join(input_dir, scenario["id"]))
                            if pattern.match(f)
                        ],
                        key=lambda x: int(pattern.match(x).group(1)),
                    )
                    for fname in fnames:
                        dur = int(pattern.match(fname).group(1))
                        exp_limits.append(dur)
                        arr = np.load(
                            os.path.join(input_dir, scenario["id"], fname)
                        )["arr_0"]
                        exp_values.append(arr)

                    exp_limits.append(scenario["config"]["t_sim"])
                    exp_limits = np.array(exp_limits)

                    # ``exp_values`` is a list that starts with a scalar 0.0.
                    # When NumPy builds an ``ndarray`` from that list together
                    # with the subsequent equal‑shaped arrays, the scalar is
                    # broadcast to the shape of the first array – exactly what
                    # the converter does.
                    exp_values = np.array(exp_values)

                    np.testing.assert_array_equal(stim["limits"][...], exp_limits)
                    np.testing.assert_allclose(stim["values"][...], exp_values)



def test_convert_to_hdf5_direct():
    """Test the converter with a direct call and a simple scenario."""
    input_dir = Path("/home/jmurugan/work/Artefacts/VT-233/vbt-converter/parametersweep_8s_3d_orientation_selective_12012026")
    output_file = Path("test_convert_output.h5")

    convert_to_hdf5(str(input_dir), str(output_file))

    # Just check that the file was created and has the expected group.
    # The detailed content validation is covered by the parameterised test above.
    with h5py.File(output_file, "r") as h5:
        assert "simulations" in h5