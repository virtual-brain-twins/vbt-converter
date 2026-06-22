from __future__ import annotations

from pathlib import Path

import numpy as np

from converters.interfaces.simulation_bundle import SimulationBundle
from converters.sections.auxilliary_files_section import AuxiliaryFileSection
from converters.sections.metadata_section import MetadataSection
from converters.sections.spiketrain_section import SpikeTrainSection


class AMUBundle(SimulationBundle):
    """
    The AMU protocol stores spikes per layer in ``<layer>.npz`` files,
    a YAML config, a few CSVs (metrics) and a binary ``record_neurons.data`` file.
    All of those are covered by the generic sections.
    """

    def __init__(self, sim_dir: Path) -> None:
        super().__init__()
        self._orig_root = sim_dir
        self._sim_id = sim_dir.name

        # ------------------------------------------------------------------
        # 1️⃣  Load YAML config (becomes metadata)
        # ------------------------------------------------------------------
        import yaml

        cfg = yaml.safe_load((sim_dir / "config.yaml").read_text())
        self.register_section("metadata", MetadataSection(cfg))

        # ------------------------------------------------------------------
        # 2️⃣  Spike trains – one per file under ``spike_times/``
        # ------------------------------------------------------------------
        spikes = {}
        spike_dir = sim_dir / "spike_times"
        for npz_file in sorted(spike_dir.glob("*.npz")):
            layer = npz_file.stem
            data = np.load(npz_file)
            spikes[layer] = (data.get("indices", None), data["times"])
        self.register_section("spike_trains", SpikeTrainSection(spikes))

        # ------------------------------------------------------------------
        # 3️⃣  Auxiliary files (logs, record_neurons, metrics, stimulus npz)
        # ------------------------------------------------------------------
        aux = {}
        for name in ("error.log", "info.log", "warning.log"):
            aux[name] = sim_dir / name
        aux["record_neurons"] = sim_dir / "record_info" / "record_neurons.data"
        aux["metrics"] = sim_dir / "metrics" / "avg_firing_rates.csv"
        # Stimulus files are a series of ``random_input_<ms>.npz``
        stim_files = list(sim_dir.glob("random_input_*.npz"))
        for f in stim_files:
            aux[f.name] = f
        self.register_section("aux_files", AuxiliaryFileSection(aux))

    # ------------------------------------------------------------------
    @property
    def sim_id(self) -> str:
        return self._sim_id

    @property
    def original_root(self) -> Path:
        return self._orig_root

    @classmethod
    def _factory_from_metadata(cls, meta: MetadataSection) -> "AMUBundle":
        # The AMU format does not embed the original directory name in the
        # metadata – we must infer it from the `sim_id` that the caller passes.
        # Hence we raise a NotImplementedError; the typical workflow is:
        #   bundle = AMUBundle(Path(...))   # for writing
        #   # For reading you just call AMUBundle.from_hdf5(...)
        raise NotImplementedError("Re‑creation from metadata only is not needed for AMU.")
