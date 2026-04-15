import random
import string
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml
# The function we want to test
from convert_direc_to_hdf import convert_to_hdf5


# -----------------------------------------------------------------
# Helper utilities – generate tiny, deterministic synthetic data
# -----------------------------------------------------------------
def _rand_string(k: int = 6) -> str:
    return "".join(random.choices(string.ascii_letters, k=k))


def _write_config_yaml(dir_path: Path, extra: dict | None = None):
    """Write a minimal config.yaml – `extra` can override or delete keys."""
    cfg = {
        "t_sim": 1000,  # ms
        "seed": 42,
        "model": "testmodel",
        "description": "synthetic test run",
    }
    if extra:
        cfg.update(extra)
    (dir_path / "config.yaml").write_text(yaml.safe_dump(cfg))


def _write_spike_file(spike_dir: Path, layer: str, indices: np.ndarray, times: np.ndarray):
    """Write a .npz file that mimics the original spike output."""
    np.savez(spike_dir / f"{layer}.npz", indices=indices, times=times)


def _write_random_input(stim_dir: Path, ms: int, values: np.ndarray):
    """Create the *_npz* stimulus file used by the converter."""
    fname = stim_dir / f"random_input_{ms}ms.npz"
    np.savez(fname, arr_0=values)


def _write_logs(log_dir: Path):
    """Create three simple log files – content does not matter for the test."""
    for name in ["error.log", "info.log", "warning.log"]:
        (log_dir / name).write_text(f"{name}: synthetic log for testing\n")


def _write_record_neurons(sim_dir: Path, neuron_counts: dict):
    """Create the pickled dict used by the converter."""
    p = sim_dir / "record_info" / "record_neurons.data"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, neuron_counts, allow_pickle=True)


def _write_metrics(sim_dir: Path, df: pd.DataFrame):
    """Create the CSV file with average firing rates."""
    (sim_dir / "metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv(sim_dir / "metrics" / "avg_firing_rates.csv", index=False)


# -----------------------------------------------------------------
# Fixtures – generate the whole input directory tree
# -----------------------------------------------------------------
@pytest.fixture
def base_input_dir(tmp_path: Path) -> Path:
    """Create a directory that contains ONE simulation sub‑folder with the *full* set of files."""
    sim_name = "sim_01"
    sim_path = tmp_path / sim_name
    sim_path.mkdir(parents=True)

    # 1️⃣ config.yaml
    _write_config_yaml(sim_path)

    # 2️⃣ spike_times (one layer with a handful of spikes)
    st_dir = sim_path / "spike_times"
    st_dir.mkdir()
    _write_spike_file(
        st_dir,
        layer="L23E",
        indices=np.arange(5, dtype=int),
        times=np.linspace(0.1, 0.5, 5),
    )

    # 3️⃣ logs
    _write_logs(sim_path)

    # 4️⃣ record_neurons
    _write_record_neurons(sim_path, {"L23E": 100, "L4I": 80})

    # 5️⃣ metrics (one row per population – we give only two rows, the rest are optional)
    df = pd.DataFrame(
        {
            "Population": ["L23E", "L4I"],
            "Avg Firing Rate Hz": [5.2, 3.1],
        }
    )
    _write_metrics(sim_path, df)

    # 6️⃣ stimulus – two random input files
    _write_random_input(sim_path, 200, np.full((10,), 0.5))
    _write_random_input(sim_path, 600, np.full((10,), 1.2))

    return tmp_path


# -----------------------------------------------------------------
# Parameterised test matrix
# -----------------------------------------------------------------
@pytest.mark.parametrize(
    "scenario,modifier,should_raise",
    [
        # ── VALID case – everything present, conversion must succeed ──
        ("valid_full", {}, None),

        # ── Missing logs – the converter opens each log file explicitly → FileNotFoundError ──
        ("missing_logs", {"remove": ["error.log", "info.log", "warning.log"]}, FileNotFoundError),

        # ── Empty spike directory – should still succeed, just no spike layers ──
        ("empty_spike", {"empty_spike_dir": True}, None),

        # ── No stimulus files – the converter still writes a stimulus with only the
        #    [0, t_sim] limits and a single zero value ──
        ("no_stimulus_files", {"remove_stim_files": True}, None),

        # ── Missing record_neurons file – code tries to open it → FileNotFoundError ──
        ("missing_record_neurons", {"remove": ["record_info/record_neurons.data"]}, FileNotFoundError),

        # ── Config without `t_sim` – the stimulus building step accesses the key → KeyError ──
        ("config_no_tsim", {"config_extra": {"t_sim": None}}, KeyError),

        # ── Corrupt spike .npz (missing 'times' field) – np.load will still succeed, but later
        #    histogram on an empty array works fine; we test that the conversion does NOT raise.
        ("corrupt_spike_file", {"corrupt_spike": True}, None),

        # ── Large number of random_input files (edge of memory) – still works ──
        ("many_random_inputs", {"many_inputs": True}, None),
    ],
    ids=lambda x: x,
)
def test_convert_to_hdf5(
        base_input_dir: Path,
        tmp_path: Path,
        scenario: str,
        modifier: dict,
        should_raise: type[BaseException] | None,
):
    """
    For each *scenario* we mutate the ``base_input_dir`` according to ``modifier``
    and then call :func:`convert_to_hdf5`.  Afterwards we perform a small set
    of sanity checks on the produced HDF5 file (if conversion succeeded).
    """
    # -----------------------------------------------------------------
    # 1️⃣  Apply the requested modifications to the synthetic tree
    # -----------------------------------------------------------------
    sim_dir = base_input_dir / "sim_01"

    # a) Remove arbitrary files
    for rel in modifier.get("remove", []):
        (sim_dir / rel).unlink(missing_ok=True)

    # b) Empty the spike_times folder
    if modifier.get("empty_spike_dir"):
        for p in (sim_dir / "spike_times").iterdir():
            p.unlink()
    # c) Delete *all* stimulus files
    if modifier.get("remove_stim_files"):
        for p in sim_dir.iterdir():
            if p.name.startswith("random_input_") and p.suffix == ".npz":
                p.unlink()
    # d) Overwrite config.yaml (e.g. remove t_sim)
    if "config_extra" in modifier:
        # Write a new config that merges the default with the requested extra dict
        cfg_path = sim_dir / "config.yaml"
        old_cfg = yaml.safe_load(cfg_path.read_text())
        new_cfg = {k: v for k, v in old_cfg.items() if v is not None}
        new_cfg.update(modifier["config_extra"])
        cfg_path.write_text(yaml.safe_dump(new_cfg))

    # e) Corrupt spike file – drop the 'times' entry (used later for histogram)
    if modifier.get("corrupt_spike"):
        spike_npz = sim_dir / "spike_times" / "L23E.npz"
        data = np.load(spike_npz, allow_pickle=True)
        # Create a new .npz that only contains indices
        np.savez(spike_npz, indices=data["indices"])

    # f) Produce a *lot* of random_input files (30 of them) – still tiny data,
    #    just to hit the loop that builds the stimulus list.
    if modifier.get("many_inputs"):
        for i in range(30):
            step = 100 + i * 50
            _write_random_input(sim_dir, step, np.full((5,), float(i)))

    # -----------------------------------------------------------------
    # 2️⃣  Run the conversion
    # -----------------------------------------------------------------
    out_h5 = tmp_path / "out.h5"

    if should_raise is not None:
        with pytest.raises(should_raise):
            convert_to_hdf5(str(base_input_dir), str(out_h5))
        return  # No further checks needed for error cases

    # No exception expected
    convert_to_hdf5(str(base_input_dir), str(out_h5))

    # -----------------------------------------------------------------
    # 3️⃣  Validate the generated HDF5
    # -----------------------------------------------------------------
    with h5py.File(out_h5, "r") as h5:
        # a) Top‑level attribute `version` must exist and be a string
        assert "version" in h5.attrs
        assert isinstance(h5.attrs["version"], (bytes, str))

        # b) The `simulations` group must contain *exactly* one subgroup (our sim)
        sims = h5["simulations"]
        assert list(sims.keys()) == ["sim_01"]

        sim_grp = sims["sim_01"]

        # c) All configuration keys from config.yaml are now attributes of the simulation group
        cfg = yaml.safe_load((sim_dir / "config.yaml").read_text())
        for key, val in cfg.items():
            # `None` gets stored as the HDF5 *null* datatype – we simply skip the check for that case
            if val is None:
                continue
            assert key in sim_grp.attrs, f"missing attribute {key}"
            # HDF5 stores numbers as numpy scalars; strings as bytes.
            stored = sim_grp.attrs[key]
            if isinstance(val, str):
                assert stored.decode() == val
            else:
                assert stored == val

        # d) Spike‑times subgroup must exist (even if empty)
        assert "spike_times" in sim_grp
        spike_grp = sim_grp["spike_times"]
        # If we removed all spike files the group will be empty – that is fine.
        # Otherwise at least one layer should be present.
        if not modifier.get("empty_spike_dir"):
            assert "L23E" in spike_grp
            layer = spike_grp["L23E"]
            assert "indices" in layer
            assert "times" in layer
            # The data we wrote must be exactly what we read back (or empty if we corrupted it)
            if modifier.get("corrupt_spike"):
                # only indices exist → times dataset should have length 0
                assert layer["times"].size == 0
            else:
                np.testing.assert_array_equal(layer["times"], np.linspace(0.1, 0.5, 5))

        # e) Log datasets – they are stored as plain strings.
        if not modifier.get("remove"):
            logs = sim_grp["logs"]
            for name in ["error.log", "info.log", "warning.log"]:
                assert name in logs
                txt = logs[name][()].decode()
                assert name.split(".")[0] in txt  # simple sanity check

        # f) `record_neurons` must be present unless we removed it.
        if not modifier.get("remove") or "record_info/record_neurons.data" not in modifier.get("remove", []):
            rec = sim_grp["record_neurons"]
            # The dtype must contain the two fields defined in the converter
            assert "label" in rec.dtype.names
            assert "value" in rec.dtype.names

        # g) Metrics – check that at least one row matches the CSV we wrote
        if "metrics" in sim_grp:
            metrics = sim_grp["metrics"]["Avg Firing Rate Hz"]
            assert "label" in metrics.dtype.names and "value" in metrics.dtype.names
            # The csv had two rows → we expect exactly two entries
            assert metrics.shape[0] == 2
            # Validate the content of the first row
            first_label = metrics[0]["label"].decode()
            first_val = metrics[0]["value"]
            assert first_label == "L23E"
            assert pytest.approx(first_val, rel=1e-6) == 5.2

        # h) Stimulus – piecewise_constant must be present, limits must start with 0 and end with t_sim
        stim = sim_grp["stimulus"]
        assert stim.attrs["type"] == "piecewise_constant"
        lim = stim["limits"][()]
        val = stim["values"][()]
        assert lim[0] == 0
        # t_sim comes from the (possibly modified) config
        expected_tsim = cfg["t_sim"]
        assert lim[-1] == expected_tsim
        # The number of limits must be len(values)+1
        assert len(lim) == len(val) + 1
        # Values must be a numpy array of objects (or scalars) – we only check shape here
        assert val.shape[0] == (len(lim) - 1)
