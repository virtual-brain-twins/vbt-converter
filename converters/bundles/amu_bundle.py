from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from converters.interfaces.simulation_bundle import SimulationBundle
from converters.sections.amu.metrics_section import MetricsSection
from converters.sections.amu.record_neurons_section import RecordNeuronsSection
from converters.sections.amu.spiketrain_section import SpikeTrainSection
from converters.sections.amu.stimulus_section import StimulusSection
from converters.sections.auxilliary_files_section import AuxiliaryFileSection
from converters.sections.metadata_section import MetadataSection
from converters.utils.common import configure_logging


class AMUBundle(SimulationBundle):
    """
    AMU bundle – a thin wrapper that mirrors the reference conversion script
    but uses the *section* abstraction.  The layout after conversion is:

    /sim_<id>/
        metadata/                → ``MetadataSection``
        logs/                    → ``AuxiliaryFileSection`` (only three log files)
        spike_trains/            → ``SpikeTrainSection``
        record_neurons/          → ``RecordNeuronsSection``
        metrics/                 → ``MetricsSection``
        stimulus/                → ``StimulusSection``
    """

    def __init__(self, sim_dir: Path, output_dir: Path = Path(".")) -> None:
        super().__init__()
        self._logger = configure_logging(
            __name__,
            log_file_name=Path(__file__).with_suffix(".log").name,
            level=logging.INFO,
        )
        self._output_dir: Path = output_dir
        self._orig_root = sim_dir
        self._sim_id = sim_dir.name

        # Load YAML config → becomes metadata
        import yaml

        cfg = yaml.safe_load((sim_dir / "config.yaml").read_text())
        self.register_section("metadata", MetadataSection(cfg))

        # Spike trains (one ``*.npz`` per layer)
        spikes: Dict[str, Tuple[np.ndarray | None, np.ndarray]] = {}
        spike_dir = sim_dir / "spike_times"
        for npz_file in sorted(spike_dir.glob("*.npz")):
            layer = npz_file.stem
            data = np.load(npz_file)
            spikes[layer] = (data.get("indices", None), data["times"])
        self.register_section("spike_trains", SpikeTrainSection(spikes))

        # Log files – keep **only** the three standard logs in an
        #     ``AuxiliaryFileSection``.
        logs: Dict[str, Path] = {}
        for log_name in ("error.log", "info.log", "warning.log"):
            log_path = sim_dir / log_name
            if log_path.is_file():
                logs[log_name] = log_path
        self.register_section("logs", AuxiliaryFileSection(logs, self._output_dir))

        # Record‑neurons (structured ``label → int`` mapping)
        rec_path = sim_dir / "record_info" / "record_neurons.data"
        rec_raw = np.load(rec_path, allow_pickle=True)
        self.register_section("record_neurons", RecordNeuronsSection(rec_raw))

        # Metrics (CSV → label → float)
        metrics_csv = sim_dir / "metrics" / "avg_firing_rates.csv"
        df = pd.read_csv(metrics_csv)
        metrics_dict = dict(zip(df["Population"], df["Avg Firing Rate Hz"]))
        self.register_section("metrics", MetricsSection(metrics_dict))

        # Stimulus – piecewise‑constant input series
        stim_type = "piecewise_constant"
        limits = [0]
        values = [0.]

        pattern = re.compile(r"random_input_(\d+)ms\.npz")
        stim_files = sorted([f for f in sim_dir.iterdir() if pattern.fullmatch(f.name)])
        for f in stim_files:
            matches = pattern.fullmatch(f.name)
            if not matches:
                continue
            limits.append(int(matches.group(1)))
            # each file contains a single array called ``arr_0``
            values.append(np.load(f)["arr_0"])

        # final limit = total simulation time (taken from the yaml config)
        limits.append(cfg["t_sim"])

        self.register_section(
            "stimulus",
            StimulusSection(
                limits=np.array(limits),
                values=np.array(values),
                stim_type=stim_type,
            ),
        )
        self._logger.info("AMU bundle created for simulation %s", self._sim_id)

    @property
    def sim_id(self) -> str:
        return self._sim_id

    @property
    def original_root(self) -> Path:
        return self._orig_root

    @classmethod
    def _factory_from_metadata(cls, meta: MetadataSection) -> "AMUBundle":
        """
        The AMU format does **not** embed the original directory name in the
        metadata – reconstruction from metadata alone is therefore impossible.
        This method exists only to satisfy the abstract API; callers should
        instantiate ``AMUBundle`` directly when they want to write a bundle.
        """
        raise NotImplementedError(
            "Re‑creation from metadata only is not needed for AMU bundles."
        )
