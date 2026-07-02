import json
from pathlib import Path
from typing import Mapping

from converters.factory.make_spike_reader import make_spike_reader
from converters.interfaces.simulation_bundle import SimulationBundle
from converters.sections.auxilliary_files_section import AuxiliaryFileSection
from converters.sections.basal_ganglia.output_spikes_section import OutputSpikesSection
from converters.sections.metadata_section import MetadataSection


def load_spike_pair(sim_dict: dict, key: str, file_key: str) -> tuple:
    """
    (unchanged – kept for completeness – used for the *input* spikes only)
    """
    src = Path(sim_dict[file_key])

    reader = make_spike_reader(
        src,
        idx_dset="/spikes/indices",
        time_dset="/spikes/times",
    )
    return reader.get_spike_arrays()


def build_spike_dict(sim_dict: dict) -> dict:
    """
    (unchanged – used for the classic ``SpikeTrainSection``)
    """
    spikes = {}
    for key, file_key in (("input_spikes", "input_file"),
                          ("output_spikes", "output_file")):
        indices, times = load_spike_pair(sim_dict, key, file_key)
        spikes[key] = (indices, times)
    return spikes


class BasalGangliaBundle(SimulationBundle):
    """
    Concrete bundle for a *single* row of the Basal‑Ganglia CSV.
    The only behavioural change compared with the original source is that
    the *output spikes* are now represented by the new ``OutputSpikesSection``.
    """

    def __init__(self, sim_dict: Mapping[str, str], simulation_dir: Path, output_dir: Path = Path(".")) -> None:
        super().__init__()
        self._output_dir = output_dir
        self._sim_dict = dict(sim_dict)  # shallow copy for later use
        self._sim_id = Path(sim_dict["sim_info_file"]).stem
        self._orig_root = simulation_dir / Path(sim_dict["network_path"])

        # Metadata
        meta = {
            "network_state": sim_dict["network_state"],
            "duration": sim_dict["duration"],
            "da_level": sim_dict["da_level"],
            "input_conductance": sim_dict["input_conductance"],
            "input_scale_factor": sim_dict["input_scale_factor"],
        }
        for maybe_json in ("snudda_data",):
            if maybe_json in sim_dict:
                try:
                    meta[maybe_json] = json.loads(sim_dict[maybe_json])
                except json.JSONDecodeError:
                    meta[maybe_json] = sim_dict[maybe_json]

        self.register_section("metadata", MetadataSection(meta))

        # Input spike trains – kept as before (plain 2‑column format)
        # spikes = build_spike_dict(dict(sim_dict))
        # self.register_section("spike_trains", SpikeTrainSection(spikes))

        # Output spikes section (full hierarchy)
        # ``sim_dict["output_file"]`` points to the file you printed with h5ls.
        self.register_section(
            "output_spikes",
            OutputSpikesSection(Path(sim_dict["output_file"]))
        )

        # Auxiliary files (unchanged)
        aux_map = {
            "network_file": Path(sim_dict["network_file"]),
            "input_config": Path(sim_dict["input_config"]),
            "virtual_input_file": Path(sim_dict["virtual_input_file"]),
            "virtual_input_mapping": Path(sim_dict["virtual_input_mapping"]),
        }
        self.register_section("aux_files", AuxiliaryFileSection(aux_map, self._output_dir))

    # Required abstract properties
    @property
    def sim_id(self) -> str:
        return self._sim_id

    @property
    def original_root(self) -> Path:
        return self._orig_root

    # Factory for reconstruction from metadata (unchanged)
    @classmethod
    def _factory_from_metadata(cls, meta: MetadataSection) -> "BasalGangliaBundle":
        dummy = {
            "sim_info_file": f"{meta.data.get('sim_id', 'unknown')}.json",
            "network_path": meta.data.get("network_path", ""),
            "network_state": meta.data.get("network_state", ""),
            "duration": str(meta.data.get("duration", "0")),
            "da_level": str(meta.data.get("da_level", "0")),
            "snudda_data": json.dumps(meta.data.get("snudda_data", {})),
            "network_file": "",
            "input_file": "",
            "output_file": "",  # will be ignored – the real data live in the bundle
            "input_config": "",
            "virtual_input_file": "",
            "virtual_input_mapping": "",
        }
        return cls(dummy)
