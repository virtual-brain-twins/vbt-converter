import pathlib
import re
import sys

import allure
import h5py
import numpy as np
import pytest

# The module we are testing -------------------------------------------------
from converters.convert_hdf_to_dsr import (
    interpolate_constant,
    get_discretized_stimulus,
    get_firing_rates,
    normalize,
    convert,
    main,
    parameters_short,
    populations,
)


# -------------------------------------------------------------------------
# Helper fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def dummy_hdf5(tmp_path):
    """
    Build a minimal but fully‑specified HDF5 file that matches the expectations
    of ``convert_hdf_to_dsr``.  The structure mirrors what ``convert_to_hdf5``
    would have produced (but we create it manually so the test stays completely
    self‑contained).

    Returns
    -------
    pathlib.Path
        Path to the created HDF5 file.
    dict
        Dictionary with a few reference values that the tests use for
        verification (e.g. the raw firing‑rate matrix before normalisation).
    """
    h5_path = tmp_path / "input.h5"

    # -----------------------------------------------------------------
    # 1️⃣ Global structure
    # -----------------------------------------------------------------
    with h5py.File(h5_path, "w") as h5:
        sims_grp = h5.create_group("simulations")
        sim = sims_grp.create_group("simA")  # a single simulation

        # -----------------------------------------------------------------
        # 2️⃣ Simulation attributes – the three scaling parameters + total time
        # -----------------------------------------------------------------
        # values are deliberately simple so that we can predict the normalised
        # value later.
        sim.attrs["scale_num_neurons"] = 1.0
        sim.attrs["scale_syn_strength"] = 2.0
        sim.attrs["scale_nmda_efficacy"] = 3.0
        sim.attrs["t_sim"] = 1000  # ms → 1 s

        # -----------------------------------------------------------------
        # 3️⃣ record_neurons – a structured array ``label`` (bytes) + ``value``
        # -----------------------------------------------------------------
        rec_arr = np.empty(len(populations), dtype=[("label", "S10"),
                                                    ("value", "i4")])
        for i, pop in enumerate(populations):
            rec_arr[i]["label"] = pop.encode()
            # give every population exactly 2 neurons (any constant >0 works)
            rec_arr[i]["value"] = 2
        sim.create_dataset("record_neurons", data=rec_arr)

        # -----------------------------------------------------------------
        # 4️⃣ spike_times – one subgroup per population, each with a ``times`` ds
        # -----------------------------------------------------------------
        spikes_grp = sim.create_group("spike_times")
        # Populate only a *few* populations with spikes – the remaining ones get
        # an empty array (the code will simply count zero spikes for them).
        for pop in populations:
            pop_grp = spikes_grp.create_group(pop)
            # give each pop a tiny, deterministic spike pattern
            if pop == "L23E":
                # three spikes, placed in three different bins
                times = np.array([0.05, 0.15, 0.95])  # seconds
            else:
                rng = np.random.default_rng(seed=42)
                times = rng.random(10)
            pop_grp.create_dataset("times", data=times)

        # -----------------------------------------------------------------
        # 5️⃣ stimulus – piecewise constant, three constant values (0, 0.5, 1.2)
        # -----------------------------------------------------------------
        stim = sim.create_group("stimulus")
        stim.attrs["type"] = "piecewise_constant"
        # limits are expressed in ms (the conversion to seconds happens inside
        # ``get_discretized_stimulus``)
        limits = np.array([0, 100, 200, 1000])  # ms
        values = np.array([0.0, 0.5, 1.2])  # one less than limits
        stim.create_dataset("limits", data=limits)
        stim.create_dataset("values", data=values)

        # -----------------------------------------------------------------
        # 6️⃣ Reference data – we compute the *raw* firing‑rate matrix here so
        #     the test can later compare it with the normalised version stored
        #     in the NPZ file.
        # -----------------------------------------------------------------
        bin_size = 0.1  # seconds (100 ms)
        t, raw_rate = get_firing_rates(
            sim["spike_times"],
            dt=bin_size,
            t0=0.0,
            t1=sim.attrs["t_sim"] / 1000.0,
            nneurons=sim["record_neurons"]
        )
        # raw_rate shape = (len(populations), nbins)
        # convert() will later transpose it, so we store the transposed version.
        reference = {
            "t": t,
            "raw_rate_T": raw_rate.T,  # shape (nbins, n_pop)
            "stimulus": get_discretized_stimulus(stim, bin_size),
            "theta": np.array([1.0, 2.0, 3.0]),
            "bin_size": bin_size,
        }

    return h5_path, reference


# -------------------------------------------------------------------------
# Test class – all test functions are methods of this single class
# -------------------------------------------------------------------------

@allure.parent_suite("VBT‑converter CI/CD")
@allure.suite("Unit tests")
@allure.sub_suite("Tests for convert_hdf_to_dsr")
@allure.description(
    "Full‑stack tests for the HDF5 → DSR conversion pipeline. "
    "Only the expected (working) scenarios are exercised – the code is "
    "assumed to be functionally correct."
)
@allure.feature("HDF5 to DSR conversion")
class TestConvertHdfToDsr:

    # -----------------------------------------------------------------
    # Helper – make the CLI call via ``main`` using a monkey‑patched argv
    # -----------------------------------------------------------------
    def _run_cli(self, input_file: pathlib.Path, output_file: pathlib.Path,
                 bin_size: float, monkeypatch):
        test_argv = [
            "convert_hdf_to_dsr.py",  # script name placeholder
            "-i", str(input_file),
            "-o", str(output_file),
            "-b", str(bin_size),
        ]
        monkeypatch.setattr(sys, "argv", test_argv)
        # ``main`` parses ``sys.argv`` internally
        with allure.step("Invoke CLI `main()`"):
            main()

    # -----------------------------------------------------------------
    # 1️⃣ Test the tiny utility ``interpolate_constant``
    # -----------------------------------------------------------------
    @allure.story("Utility – interpolate_constant")
    @allure.title("interpolate_constant returns correct piecewise‑constant values")
    @allure.severity(allure.severity_level.TRIVIAL)
    def test_interpolate_constant(self):
        xp = np.array([0, 1, 2])
        yp = np.array([-1, 0, 1, 2])  # len(yp) == len(xp) + 1
        x = np.array([-0.5, 0.2, 0.9, 1.5, 2.3])

        # -------------------------------------------------------------
        # Without extrapolation – behaviour is defined for x in [xp[0], xp[-1])
        # -------------------------------------------------------------
        with allure.step("Without extrapolation (default)"):
            y_no_ex = interpolate_constant(x, xp, yp, extrapolate=False)
            expected = np.array([-1, 0, 0, 1, 2])
            np.testing.assert_array_equal(y_no_ex, expected)

        # -------------------------------------------------------------
        # With extrapolation – the first and last values are duplicated
        # -------------------------------------------------------------
        with allure.step("With extrapolation"):
            y_ex = interpolate_constant(x, xp, yp, extrapolate=True)
            # After extrapolation yp becomes [-1, -1, 0, 1, 2, 2]
            expected_ex = np.array([-1, -1, -1, 0, 1])
            np.testing.assert_array_equal(y_ex, expected_ex)

    # -----------------------------------------------------------------
    # 2️⃣ Test the stimulus discretisation helper
    # -----------------------------------------------------------------
    @allure.story("Utility – get_discretized_stimulus")
    @allure.title("Piecewise‑constant stimulus is converted to a regular array")
    @allure.severity(allure.severity_level.MINOR)
    def test_get_discretized_stimulus(self, dummy_hdf5):
        _, reference = dummy_hdf5
        # The reference fixture already generated the expected stimulus array.
        expected = reference["stimulus"]
        # Verify shape and content directly.
        assert isinstance(expected, np.ndarray)
        # The shape should be (nbins,) – in our case 10 (1 s / 0.1 s)
        assert expected.shape == (10,)

        expected_manual = np.array(
            [0.0] + [0.5] * 1 + [1.2] * 8,
            dtype=float
        )
        np.testing.assert_allclose(expected, expected_manual)

    # -----------------------------------------------------------------
    # 3️⃣ Test the firing‑rate extraction helper
    # -----------------------------------------------------------------
    @allure.story("Utility – get_firing_rates")
    @allure.title("Firing rates are correctly computed per population & bin")
    @allure.severity(allure.severity_level.NORMAL)
    def test_get_firing_rates(self, dummy_hdf5):
        h5_path, reference = dummy_hdf5
        with h5py.File(h5_path, "r") as h5:
            sim = h5["simulations"]["simA"]
            t, fr = get_firing_rates(
                sim["spike_times"],
                dt=reference["bin_size"],
                t0=0.0,
                t1=sim.attrs["t_sim"] / 1000.0,
                nneurons=sim["record_neurons"],
            )
        # -------------------------------------------------------------
        # Verify raw timings
        # -------------------------------------------------------------
        assert np.allclose(t, reference["t"])

        # -------------------------------------------------------------
        # Only population L23E has spikes – verify its rates
        # -------------------------------------------------------------
        idx_l23e = populations.index("L23E")
        # three spikes: 0.05 s in bin0, 0.15 s in bin1, 0.95 s in bin9
        # each bin width = 0.1 s, neurons per pop = 2
        expected_rates = np.zeros_like(fr[idx_l23e])
        expected_rates[0] = 1 / (2 * 0.1)  # 5 Hz
        expected_rates[1] = 1 / (2 * 0.1)  # 5 Hz
        expected_rates[9] = 1 / (2 * 0.1)  # 5 Hz
        np.testing.assert_allclose(fr[idx_l23e], expected_rates)

    # -----------------------------------------------------------------
    # 4️⃣ Full conversion – file creation and content sanity checks
    # -----------------------------------------------------------------
    @allure.story("End‑to‑end conversion")
    @allure.title("convert() creates a correctly normalised NPZ archive")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_full_conversion(self, dummy_hdf5, tmp_path):
        h5_path, reference = dummy_hdf5
        out_npz = tmp_path / "output.npz"

        # -------------------------------------------------------------
        # Run the conversion
        # -------------------------------------------------------------
        with allure.step("Call convert(...)"):
            convert(str(h5_path), str(out_npz), reference["bin_size"])

        # -------------------------------------------------------------
        # Load the resulting archive and perform sanity checks
        # -------------------------------------------------------------
        with allure.step("Load NPZ file and inspect contents"):
            data = np.load(out_npz, allow_pickle=True)

            # ----- Shapes -------------------------------------------------
            assert data["x"].shape == (1, len(reference["t"]), len(populations))
            assert data["u"].shape == (1, len(reference["t"]), 1)
            assert data["theta"].shape == (1, 3)  # three parameters

            # ----- Names --------------------------------------------------
            np.testing.assert_array_equal(data["names_x"], np.array(populations))
            np.testing.assert_array_equal(data["names_theta"], np.array(parameters_short))

            # ----- dt -----------------------------------------------------
            assert data["dt"] == pytest.approx(reference["bin_size"])

            # ----- Stimulus ------------------------------------------------
            # ``u`` is stored as (npoints, nt, 1) – squeeze the last dim
            u_arr = data["u"].squeeze()
            np.testing.assert_allclose(u_arr, np.array([-2.549822, -1.287534] + [0.4796695271974878] * 8), rtol=1e-6,
                                       atol=1e-8)

            # ----- Theta (parameters) ------------------------------------
            # ``theta`` is normalised – we check that the *raw* (pre‑normalised)
            # values are stored in ``mu_theta`` and ``sd_theta``.
            raw_theta = reference["theta"]
            mu_theta = data["mu_theta"]
            sd_theta = data["sd_theta"]
            np.testing.assert_allclose(raw_theta,
                                       raw_theta * (sd_theta + 0) + mu_theta,
                                       atol=1e-12)  # sanity – mean/std are correct

            # ----- Firing‑rates --------------------------------------------
            # The raw, non‑normalised matrix is stored in ``mu_x`` / ``sd_x``.
            # We verify that after de‑normalising we obtain the reference matrix.
            x_norm = data["x"][0]  # (nt, n_pop)
            mu_x = data["mu_x"]
            sd_x = data["sd_x"]
            # reverse normalisation
            x_denorm = x_norm * sd_x + mu_x
            np.testing.assert_allclose(x_denorm,
                                       reference["raw_rate_T"],
                                       rtol=1e-6, atol=1e-8)

    # -----------------------------------------------------------------
    # 5️⃣ CLI entry point – make sure ``main`` wires arguments correctly
    # -----------------------------------------------------------------
    @allure.story("CLI – command line interface")
    @allure.title("Running the module as a script creates the expected NPZ file")
    @allure.severity(allure.severity_level.NORMAL)
    def test_cli_invocation(self, dummy_hdf5, tmp_path, monkeypatch):
        h5_path, reference = dummy_hdf5
        out_npz = tmp_path / "cli_output.npz"

        # -------------------------------------------------------------
        # Simulate ``python -m convert_hdf_to_dsr -i <in> -o <out> -b <bin>`` 
        # -------------------------------------------------------------
        self._run_cli(h5_path, out_npz, reference["bin_size"], monkeypatch)

        # -------------------------------------------------------------
        # Very light validation – just that the file exists and contains key attrs
        # -------------------------------------------------------------
        assert out_npz.is_file(), "NPZ file not created by CLI"
        data = np.load(out_npz, allow_pickle=True)
        assert "theta" in data.files
        assert "x" in data.files
        assert "u" in data.files
        # The version information lives only in the HDF5 input, but we can still
        # check that the delta‑time stored matches the request.
        assert data["dt"] == pytest.approx(reference["bin_size"])

    # -----------------------------------------------------------------
    # 6️⃣ Normalisation helper – sanity check on statistics
    # -----------------------------------------------------------------
    @allure.story("Utility – normalize")
    @allure.title("normalize returns correctly centred and scaled arrays")
    @allure.severity(allure.severity_level.TRIVIAL)
    def test_normalize(self):
        # 2‑D example where we know the expected mean/std
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_norm, mu, sigma = normalize(X, axis=0)  # column‑wise

        # Expected: column means = [2, 3]; std = [1, 1]
        np.testing.assert_allclose(mu, np.array([2.0, 3.0]))
        np.testing.assert_allclose(sigma, np.array([1.0, 1.0]))
        # After normalisation we expect [[-1, -1], [1, 1]]
        np.testing.assert_allclose(X_norm, np.array([[-1, -1], [1, 1]]))

    # -----------------------------------------------------------------
    # 7️⃣ Regular‑expression used for stimulus file names (mirrored from the
    #     companion converter).  Even though it is not used directly in the
    #     module under test, keeping the test mirrors the original test‑suite
    # -----------------------------------------------------------------
    @allure.story("Utility – stimulus filename regex")
    @allure.title("Regex matches only valid stimulus filenames")
    @allure.severity(allure.severity_level.TRIVIAL)
    def test_stimulus_regex(self):
        pattern = re.compile(r"random_input_(\d+)ms\.npz")
        good = ["random_input_10ms.npz", "random_input_250ms.npz"]
        bad = ["randominput_10ms.npz", "random_input_10ms.txt", "random_input_ms.npz"]

        for g in good:
            with allure.step(f"Accept good pattern: {g}"):
                m = pattern.match(g)
                assert m is not None
                assert int(m.group(1)) > 0

        for b in bad:
            with allure.step(f"Reject bad pattern: {b}"):
                assert pattern.match(b) is None
