import pathlib
import pickle
import re
import sys

# ------------------------------------------------------------
# Allure imports & configuration
# ------------------------------------------------------------
import allure
import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from converters.convert_direc_to_hdf import convert_to_hdf5, main


# ------------------------------------------------------------
# Helper: create a dummy simulation directory tree inside a temporary path
# ------------------------------------------------------------
def _create_dummy_simulation(root: pathlib.Path, name: str):
    """
    Build the following structure (all paths are relative to `root`):

    name/
        config.yaml
        spike_times/
            L23.npz
        error.log
        info.log
        warning.log
        record_info/
            record_neurons.data
        metrics/
            avg_firing_rates.csv
        random_input_100ms.npz
        random_input_200ms.npz
    """
    sim_dir = root / name
    sim_dir.mkdir(parents=True)

    # ---------------------- config.yaml ----------------------
    config = {
        "model": "test_model",
        "t_sim": 350,  # total simulation time (ms)
        "seed": 42,
    }
    (sim_dir / "config.yaml").write_text(yaml.safe_dump(config))

    # ------------------------ spike_times ------------------------
    spike_dir = sim_dir / "spike_times"
    spike_dir.mkdir()
    # create a single layer file "L23.npz"
    indices = np.arange(0, 10, dtype=int)
    times = np.linspace(0, 100, 10)
    np.savez(spike_dir / "L23.npz", indices=indices, times=times)

    # --------------------------- logs ---------------------------
    for log_name, content in [
        ("error.log", "error line 1\nerror line 2"),
        ("info.log", "info line 1\ninfo line 2"),
        ("warning.log", "warning line 1\nwarning line 2"),
    ]:
        (sim_dir / log_name).write_text(content)

    # ----------------------- record_info -----------------------
    rec_dir = sim_dir / "record_info"
    rec_dir.mkdir()
    # store a dict {label: value}
    record_dict = {"neuron_A": 1, "neuron_B": 2, "neuron_C": 3}
    file_path = rec_dir / "record_neurons.data"
    with open(file_path, "wb") as f:
        pickle.dump(record_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------- metrics --------------------------
    met_dir = sim_dir / "metrics"
    met_dir.mkdir()
    df = pd.DataFrame({
        "Population": ["pop1", "pop2"],
        "Avg Firing Rate Hz": [5.0, 12.7],
    })
    df.to_csv(met_dir / "avg_firing_rates.csv", index=False)

    # ------------------------ stimulus ------------------------
    # create two piece‑wise constant input files
    for ms, value in [(100, 0.5), (200, 1.2)]:
        arr = np.full((), value)  # arbitrary shape, just needs to be loadable
        np.savez(sim_dir / f"random_input_{ms}ms.npz", arr_0=arr)

    return config  # return config for later checks


# ----------------------------------------------------------------------
# Fixture that builds a temporary input directory with one simulation
# ----------------------------------------------------------------------
@pytest.fixture
def dummy_input_dir(tmp_path):
    """
    Returns the path to a temporary directory that contains one dummy simulation.
    """
    input_dir = tmp_path / "input"
    config = _create_dummy_simulation(input_dir, "sim1")
    # also add a second “empty” directory to verify the script skips non‑simulation
    # folders that are missing expected sub‑folders – this also checks robustness.
    (input_dir / "empty_folder").mkdir()
    return input_dir, config


# ----------------------------------------------------------------------
# Test class – all test functions are now methods of this class.
# ----------------------------------------------------------------------
@allure.parent_suite("VBT‑converter CI/CD")
@allure.suite("Unit tests")
@allure.sub_suite("Tests for convert_to_hdf5.py")
@allure.description("Unit tests for convert_to_hdf5.py")
@allure.feature("HDF5 conversion from the parameter sweep directory")
class TestConvertToHdf5:
    # --------------------------------------------------------------
    # Test the core conversion function
    # --------------------------------------------------------------
    @allure.story("Basic conversion of a complete simulation directory")
    @allure.title("convert_to_hdf5 creates a correct HDF5 file from a valid simulation directory")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_convert_to_hdf5_basic(self, dummy_input_dir, tmp_path):
        input_dir, config = dummy_input_dir

        out_file = tmp_path / "output_dir" / "out.h5"
        out_file.parent.mkdir(parents=True, exist_ok=True)  # ensure target folder exists

        # ------------------------------------------------------------------
        # Call the function under test
        # ------------------------------------------------------------------
        with allure.step("Run convert_to_hdf5"):
            convert_to_hdf5(str(input_dir), str(out_file))

        # ------------------------------------------------------------------
        # Basic sanity checks on the output file
        # ------------------------------------------------------------------
        with allure.step("Verify output file existence"):
            assert out_file.is_file(), "Output HDF5 file was not created"

        with h5py.File(out_file, "r") as h5:
            # ---- global attribute ----
            with allure.step("Check global version attribute"):
                assert h5.attrs["version"] == "0.1"

            # ---- simulation group ----
            sims = h5["simulations"]
            with allure.step("Validate simulation group name"):
                assert "sim1" in sims, "Simulation group missing"
            sim = sims["sim1"]

            # ---- config attributes ----
            with allure.step("Validate stored config attributes"):
                for k, v in config.items():
                    assert k in sim.attrs
                    stored = sim.attrs[k]
                    if isinstance(stored, bytes):
                        stored = stored.decode()
                    assert stored == v

            # ---- spike times ----
            with allure.step("Validate spike‑time datasets"):
                spike_grp = sim["spike_times"]
                assert "L23" in spike_grp
                layer = spike_grp["L23"]
                np.testing.assert_array_equal(layer["indices"][()], np.arange(0, 10))
                np.testing.assert_array_equal(layer["times"][()], np.linspace(0, 100, 10))

            # ---- logs ----
            with allure.step("Validate log datasets"):
                logs = sim["logs"]
                for log_name, text in [
                    ("error.log", "error line 1\nerror line 2"),
                    ("info.log", "info line 1\ninfo line 2"),
                    ("warning.log", "warning line 1\nwarning line 2"),
                ]:
                    stored = logs[log_name][()]
                    if isinstance(stored, bytes):
                        stored = stored.decode()
                    assert stored == text

            # ---- record_neurons ----
            with allure.step("Validate record_neurons structured array"):
                rec = sim["record_neurons"]
                labels = rec["label"][()].astype(str).tolist()
                values = rec["value"][()].tolist()
                expected_labels = ["neuron_A", "neuron_B", "neuron_C"]
                expected_values = [1, 2, 3]
                assert sorted(labels) == sorted(expected_labels)
                assert sorted(values) == sorted(expected_values)

            # ---- metrics ----
            with allure.step("Validate metrics datasets"):
                metrics_grp = sim["metrics"]
                assert "Avg Firing Rate Hz" in metrics_grp
                metric_ds = metrics_grp["Avg Firing Rate Hz"]
                metric_labels = metric_ds["label"][()].astype(str).tolist()
                metric_vals = metric_ds["value"][()].tolist()
                assert metric_labels == ["pop1", "pop2"]
                assert metric_vals == [5.0, 12.7]

            # ---- stimulus (piecewise constant) ----
            with allure.step("Validate stimulus piecewise‑constant representation"):
                stim = sim["stimulus"]
                assert stim.attrs["type"] == "piecewise_constant"
                limits = stim["limits"][()].tolist()
                values = stim["values"][()].tolist()

                # limits = [0, 100, 200, t_sim]
                assert limits == [0, 100, 200, config["t_sim"]]

                # values[0] is the initial zero‑value array, then the two arrays we saved
                assert isinstance(values[0], float) and np.allclose(values[0], 0.0)
                assert isinstance(values[1], float) and np.allclose(values[1], 0.5)
                assert isinstance(values[2], float) and np.allclose(values[2], 1.2)

    # --------------------------------------------------------------
    # Test the CLI entry point (i.e. `python -m convert_to_hdf5 ...`)
    # --------------------------------------------------------------
    @allure.story("CLI invocation")
    @allure.title("Running the module as a script creates the expected HDF5 file")
    @allure.severity(allure.severity_level.NORMAL)
    def test_cli_invocation(self, dummy_input_dir, tmp_path, monkeypatch):
        input_dir, _ = dummy_input_dir
        out_file = tmp_path / "cli_out.h5"

        # Simulate command‑line arguments
        test_args = [
            "convert_to_hdf5.py",  # script name (ignored by argparse)
            "-i", str(input_dir),
            "-o", str(out_file),
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        # Run the CLI entry point; it should call `convert_to_hdf5`
        with allure.step("Execute main() with patched argv"):
            main()

        # Very small sanity check – the file must exist and contain the version attribute.
        with allure.step("Validate CLI‑produced file"):
            assert out_file.is_file()
            with h5py.File(out_file, "r") as h5:
                assert h5.attrs["version"] == "0.1"

    # --------------------------------------------------------------
    # Ensure the script fails gracefully when required files are missing
    # --------------------------------------------------------------
    @allure.story("Error handling – missing required files")
    @allure.title("Missing config.yaml raises FileNotFoundError")
    @allure.severity(allure.severity_level.MINOR)
    def test_missing_config_raises(self, tmp_path):
        """
        If a simulation directory does not contain a `config.yaml`,
        the script should raise a `FileNotFoundError`.
        """
        # Create a simulation folder without a config file
        input_dir = tmp_path / "input_dir"
        sim = input_dir / "bad_sim"
        sim.mkdir(parents=True, exist_ok=True)
        # provide the other required sub‑folders so that the failure comes from the missing config
        (sim / "spike_times").mkdir()
        (sim / "record_info").mkdir()
        (sim / "metrics").mkdir()
        (sim / "logs").mkdir()

        out_file = tmp_path / "bad_out.h5"
        out_file.parent.mkdir(parents=True, exist_ok=True)  # ensure target folder exists

        with allure.step("Expect FileNotFoundError when config.yaml is absent"):
            with pytest.raises(FileNotFoundError):
                convert_to_hdf5(str(input_dir), str(out_file))

    # --------------------------------------------------------------
    # Helper: make sure the regular‑expression used for stimulus files works
    # --------------------------------------------------------------
    @allure.story("Utility – stimulus filename regex")
    @allure.title("Stimulus filename regular expression matches only valid patterns")
    @allure.severity(allure.severity_level.TRIVIAL)
    def test_stimulus_regex(self):
        pattern = re.compile(r"random_input_(\d+)ms\.npz")
        good = ["random_input_10ms.npz", "random_input_250ms.npz"]
        bad = ["randominput_10ms.npz", "random_input_10ms.txt", "random_input_ms.npz"]

        for f in good:
            with allure.step(f"Validate good pattern: {f}"):
                m = pattern.match(f)
                assert m is not None
                assert int(m.group(1)) > 0

        for f in bad:
            with allure.step(f"Validate bad pattern is rejected: {f}"):
                assert pattern.match(f) is None
