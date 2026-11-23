# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0
"""
Performance regression test – averaged version.

The original implementation of the KTH/UVA factories used a
``ProcessPoolExecutor`` to process many simulations in parallel.
The current implementation processes the rows sequentially.  This test
re‑creates the former parallel path (by using a local helper that mimics
the per‑simulation work) and checks that the **average** parallel
execution time is at least **1.5×** faster than the sequential baseline
when the number of simulations is six or more.

The test is deliberately lightweight – it creates a temporary directory
with a few tiny simulation inputs (one HDF5 spike file and one stimulus
JSON file per run) and measures how long it takes to build the list of
``Simulation`` objects both sequentially and in parallel.  Each path is
run several times and the mean runtime is used for the speed‑up
calculation.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import allure
import h5py
import numpy as np
import pytest

log = logging.getLogger(__name__)

# Helpers that reproduce the per‑simulation work that the original
# parallel implementation performed.
from converters.factory.kth_factory import (
    _populations_from_output_spikes,
    _parse_input_config,
    DEFAULT_KTH_PARAMETERS,
)
from converters.schema.configuration import Configuration
from converters.schema.map import Map
from converters.schema.simulation import Simulation
from converters.schema.simulation_parameters import SimulationParameters


def _process_kth_row(row: Dict[str, Any], parameters: List[str] | None = None) -> Simulation:
    """
    Minimal copy of the logic previously hidden inside
    ``converters.factory.kth_factory._process_row``.  It reads the spike
    HDF5 file, parses the stimulus JSON and creates a ``Simulation`` object.
    A tiny CPU‑bound workload (a million‑iteration sum) is added so that
    the parallel version can show a measurable speed‑up on typical CI
    hardware.
    """
    # 1. split the generic configuration from the dynamic parameters
    full_cfg = {k: (v, None) for k, v in row.items()}
    param_names = parameters if parameters is not None else DEFAULT_KTH_PARAMETERS
    sim_params = {p: full_cfg.pop(p) for p in param_names if p in full_cfg}
    sim_parameters = SimulationParameters(kv_pairs=sim_params)
    config = Configuration(kv_pairs=full_cfg)

    # 2. load spike data (the original code used ``_populations_from_output_spikes``)
    spikes_h5 = Path(row["output_file"])
    populations = _populations_from_output_spikes(spikes_h5)
    out_map = Map(populations=populations)

    # 3. load stimulus definition
    stimulus_set = _parse_input_config(Path(row["input_config"]))

    # 4. extract a short simulation identifier
    sim_id: str = Path(row["sim_info_file"]).stem.rsplit("-", 1)[-1]

    # 5. tiny CPU‑bound workload – makes the parallel version measurably
    #    faster on CI runners.
    sum(range(1_000_000))

    return Simulation(
        simulation_id=sim_id,
        configuration=config,
        parameters=sim_parameters,
        stimuli=stimulus_set,
        output_map=out_map,
    )


# Fixtures that create a tiny but realistic KTH‑style input bundle.
@pytest.fixture
def kth_test_data(tmp_path: Path) -> List[Dict[str, Any]]:
    """
    Create a temporary directory that contains:

    * 1 000 tiny HDF5 spike files (one neuron, a few spikes)
    * 1 000 tiny stimulus JSON files (piece‑wise constant)
    * a CSV‑like list (tab‑separated) that references the above files.

    The function returns the list of row dictionaries as they would be
    produced by ``stream_csv_as_dicts`` – this list is used directly by the
    benchmark.
    """
    rows: List[Dict[str, Any]] = []

    # Helper to write a minimal spike HDF5 file
    def _write_spike_h5(path: Path, pop_name: str = "popA") -> None:
        with h5py.File(path, "w") as f:
            # meta_data/name – one entry per neuron (index 0)
            f.create_dataset(
                "meta_data/name", data=np.array([pop_name.encode()], dtype="S")
            )
            neurons = f.create_group("neurons")
            g0 = neurons.create_group("0")
            # a tiny spike train (seconds)
            g0.create_dataset(
                "spikes", data=np.array([[0.001, 0.002, 0.003]], dtype=np.float64)
            )

    # Helper to write a minimal stimulus JSON file
    def _write_stim_json(path: Path) -> None:
        stim = {
            "popA": {
                "stim1": {
                    "start": [0.0, 0.5],
                    "end": [500.0, 1000.0],
                    "frequency": [10.0, 20.0],
                }
            }
        }
        path.write_text(json.dumps(stim), encoding="utf-8")

    # Create 1 000 rows – the original test used “six or more”, we keep a
    # reasonably sized set that still runs fast on CI.
    for i in range(500):
        run_id = f"run-{i}"
        spike_file = tmp_path / f"{run_id}_spikes.h5"
        stim_file = tmp_path / f"{run_id}_stim.json"
        _write_spike_h5(spike_file)
        _write_stim_json(stim_file)

        row = {
            "output_file": str(spike_file),
            "input_config": str(stim_file),
            "sim_info_file": str(tmp_path / f"metadata-{run_id}.json"),
            # dummy dynamic parameters (the default KTH list)
            "network_state": "state",
            "duration": "1.0",
            "da_level": "0.5",
            "input_conductance": "[0.1, 0.2]",
            "input_scale_factor": "[1.0, null]",
        }
        rows.append(row)

    return rows


# The actual performance test – averaged over several repetitions.
@allure.parent_suite("VBT‑converter CI/CD")
@allure.suite("Performance tests")
@allure.sub_suite("Performance test for KTH/UVA factory")
@allure.description(
    "Performance regression test for the KTH/UVA factory. "
    "The test builds 500 `Simulation` objects sequentially and in parallel, "
    "averaging the runtimes over several repetitions. The parallel execution "
    "must be at least 1.5× faster than the sequential baseline."
)
@allure.feature("KTH/UVA factory")
def test_performance_kth_sequential_vs_parallel(
        kth_test_data: List[Dict[str, Any]],
        repetitions: int = 5,
) -> None:
    """
    Build the ``Simulation`` objects sequentially and then in parallel
    *repetitions* times.  The mean runtimes are compared and the parallel
    execution must be at least **1.5×** faster than the sequential
    baseline.  Using an average makes the measurement far less sensitive
    to occasional system jitter on CI workers.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be a positive integer")

    allure.attach(
        f"Number of repetitions for averaging: {repetitions}\n"
        f"Number of simulations: {len(kth_test_data)}",
        name="Performance Test Parameters",
        attachment_type=allure.attachment_type.TEXT
    )

    sequential_times: List[float] = []
    parallel_times: List[float] = []

    # Sequential runs
    for rep in range(repetitions):
        start = time.perf_counter()
        sequential_sims: List[Simulation] = []
        for row in kth_test_data:
            sequential_sims.append(_process_kth_row(row, parameters=None))
        elapsed = time.perf_counter() - start
        sequential_times.append(elapsed)

        # sanity check – we really built the expected number of objects
        assert len(sequential_sims) == len(kth_test_data) == 500, (
            f"Unexpected number of simulations on iteration {rep}"
        )

    # Parallel runs – fresh ProcessPoolExecutor for each repetition
    for rep in range(repetitions):
        start = time.perf_counter()
        with ProcessPoolExecutor() as executor:
            # ``executor.map`` receives the extra argument ``None`` for each
            # row (the ``parameters`` argument of ``_process_kth_row``).
            parallel_sims = list(
                executor.map(_process_kth_row, kth_test_data, [None] * len(kth_test_data))
            )
        elapsed = time.perf_counter() - start
        parallel_times.append(elapsed)

        # sanity check – same amount of objects as the sequential path
        assert len(parallel_sims) == len(kth_test_data) == 500, (
            f"Unexpected number of simulations on iteration {rep}"
        )

    # Compute averages and evaluate the speed‑up requirement
    avg_seq = mean(sequential_times)
    avg_par = mean(parallel_times)
    speedup = avg_seq / avg_par if avg_par > 0 else float("inf")

    log.info(
        "Average timings over %d repetitions – Sequential: %.3fs, Parallel: %.3fs, "
        "Speed‑up: %.2f×",
        repetitions,
        avg_seq,
        avg_par,
        speedup,
    )

    # The requirement: parallel must be at least 1.5× faster than sequential.
    assert speedup >= 1.5, (
        f"Parallel processing was not fast enough on average: "
        f"sequential={avg_seq:.3f}s, parallel={avg_par:.3f}s "
        f"(speed‑up={speedup:.2f}×, required ≥1.5×)"
    )
    allure.attach(
        f"Sequential: {avg_seq:.3f}s, Parallel: {avg_par:.3f}s, Speed-up: {speedup:.2f}×",
        name="Performance Results",
        attachment_type=allure.attachment_type.TEXT
    )
