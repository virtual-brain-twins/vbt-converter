import argparse
import json
from typing import Any

import h5py
import numpy as np

from .protocol_exporter import ProtocolExporter


class BasalGangliaExporter(ProtocolExporter):
    """
    Export the master HDF5 into a *single* ``.npz`` per simulation that contains
    the raw spike‑train arrays for every population.  The format is deliberately
    minimal because downstream code usually wants the original per‑population
    ``(indices, times)`` arrays.

    Output file layout (npz keys):

        <pop>_indices : int[N]   (may be absent → omitted)
        <pop>_times   : float[N] (seconds)
        metadata      : JSON‑encoded string with all scalar metadata
    """

    def extract_one_simulation(self, sim_id: str, sim_group: h5py.Group) -> Any:
        pass

    protocol_name = "basal-ganglia"
    output_suffix = ".basal.npz"

    @classmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        # No extra arguments needed for now, but the hook is kept
        pass

    # TO DO: Need to be fully implemented
    def export_one_simulation(self, sim_id: str, sim_group: h5py.Group) -> None:
        # Load metadata
        meta_grp = self._read_section(sim_group, "metadata")
        # Convert everything to JSON (easy round‑trip)
        meta_dict = {k: meta_grp.attrs[k] for k in meta_grp.attrs}
        meta_json = json.dumps(meta_dict, default=str)

        # Load spike‑train section
        spikes_sec = self._read_section(sim_group, "spike_trains")

        # Prepare the dictionary that will be handed to ``np.savez``.
        out_dict: dict[str, Any] = {"metadata": meta_json}

        for pop in spikes_sec.keys():
            pop_grp = spikes_sec[pop]
            times = self._as_numpy(pop_grp["times"])
            out_dict[f"{pop}_times"] = times

            if "indices" in pop_grp:
                idx = self._as_numpy(pop_grp["indices"])
                out_dict[f"{pop}_indices"] = idx

        # Write the file
        out_path = self.out_dir / f"{sim_id}{self.output_suffix}"
        np.savez(out_path, **out_dict)
        self.log.info("Wrote Basal‑Ganglia archive %s (%d populations)", out_path, len(spikes_sec))
