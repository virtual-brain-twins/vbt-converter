import argparse
import os
import re
from typing import cast, Iterable

import h5py
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def convert_to_hdf5(input_dir, output_file):
    with h5py.File(output_file, "w") as h:
        h.attrs['version'] = '0.1'
        sims_grp = h.create_group("simulations")
        for directory in tqdm(sorted(cast(Iterable[str], os.listdir(input_dir)))):
            full_path = str(os.path.join(input_dir, directory))
            if not os.path.isdir(full_path) or not os.listdir(full_path):
                continue  # not a folder → ignore | empty folder → skip
            sim_grp = sims_grp.create_group(directory)

            # Configuration and metadata
            config_path = os.path.join(full_path, "config.yaml")
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
                for key, val in config_dict.items():
                    sim_grp.attrs[key] = val

            # Spikes
            spike_dir = os.path.join(full_path, "spike_times")
            spike_grp = sim_grp.create_group("spike_times")
            for fname in os.listdir(spike_dir):
                layer, ext = os.path.splitext(fname)
                data = np.load(os.path.join(spike_dir, fname))

                layer_grp = spike_grp.create_group(layer)
                layer_grp.create_dataset("indices", data=data['indices'].astype(int))
                layer_grp.create_dataset("times", data=data['times'])

            # Logs
            log_grp = sim_grp.create_group("logs")
            for log_name in ['error.log', 'info.log', 'warning.log']:
                log_file = os.path.join(full_path, log_name)
                with open(log_file, "r") as fh:
                    log_text = fh.read()
                log_grp.create_dataset(log_name, data=log_text)

            # Record info
            data = np.load(os.path.join(full_path, "record_info/record_neurons.data"), allow_pickle=True)
            dtype = np.dtype([("label", h5py.string_dtype()), ("value", int)])
            arr = np.empty(len(data), dtype=dtype)
            arr["label"] = np.array(list(data.keys()))
            arr["value"] = np.array(list(data.values()))
            sim_grp.create_dataset("record_neurons", data=arr)

            # Metrics
            df = pd.read_csv(os.path.join(full_path, "metrics/avg_firing_rates.csv"))
            metrics_grp = sim_grp.create_group("metrics")
            dtype = np.dtype([("label", h5py.string_dtype()), ("value", np.float64)])
            arr = np.empty(len(df), dtype=dtype)
            arr["label"] = df['Population']
            arr["value"] = df['Avg Firing Rate Hz']
            metrics_grp.create_dataset("Avg Firing Rate Hz", data=arr)

            # Input
            stim_grp = sim_grp.create_group("stimulus")
            stim_grp.attrs['type'] = 'piecewise_constant'
            limits = [0]
            values = [0.]
            pattern = re.compile(r"random_input_(\d+)ms\.npz")
            fnames = sorted([f for f in os.listdir(str(os.path.join(full_path))) if pattern.match(f)])
            for fname in fnames:
                limits.append(int(pattern.match(fname).group(1)))
                values.append(np.load(os.path.join(full_path, fname))['arr_0'])
            limits.append(config_dict['t_sim'])
            stim_grp.create_dataset("limits", data=np.array(limits))
            stim_grp.create_dataset("values", data=np.array(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    args = parser.parse_args()

    convert_to_hdf5(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
