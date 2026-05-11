#!/usr/bin/env python
"""
convert_direc_to_npz.py
--------------------
A fast, parallel, memory‑efficient conversion from a Neuro‑Modeling sweep
directory (workflow‑1 output) directly to the NPZ (or HDF5) format expected by
workflow‑2.

Additional feature
~~~~~~~~~~~~~~~~~~
* Intermediate HDF5 file that stores the full simulation
  information (spikes, logs, stimulus, record info, config …) while the
  fast NPZ conversion proceeds.  The HDF5 file is written inside the worker
  process (`_process_simulation`) and is protected by a process‑wide lock so
  that several workers can safely append their groups.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from multiprocessing import Lock, get_context

# Global objects used by the workers (initialized in the executor)
ctx = get_context("spawn")
_hdf5_lock: Optional[Lock] = None          # will be set by the executor initializer
_intermediate_h5_path: Optional[Path] = None   # path of the intermediate HDF5 file


def _worker_init(lock_: Lock, h5_path: Optional[str]) -> None:
    """
    Initializer for each worker process.  Stores a reference to a lock and the
    intermediate HDF5 path in module‑level globals so that `_process_simulation`
    can safely write to the file.
    """
    global _hdf5_lock, _intermediate_h5_path
    _hdf5_lock = lock_
    _intermediate_h5_path = Path(h5_path) if h5_path is not None else None


# Configuration dataclass
@dataclass
class ConversionConfig:
    """All knobs that a user may want to tweak."""
    # Directories / files
    input_dir: Path
    output_file: Path
    intermediate_hdf5: Optional[Path] = None
    output_format: str = "npz"          # "npz" or "hdf5"
    compression: str = "gzip"           # "gzip" (npz) or "blosc" (zarr) – also used for HDF5
    compression_opts: int = 4           # gzip level or blosc clevel
    skip_missing: bool = False          # fail hard (False) or ignore missing data

    # Simulation‑wise parameters
    bin_size_s: float = 0.001           # dt in seconds (default 1 ms)
    populations: List[str] = field(
        default_factory=lambda: [
            "L1VIP",
            "L23E", "L23PV", "L23SST", "L23VIP",
            "L4E", "L4PV", "L4SST", "L4VIP",
            "L5E", "L5PV", "L5SST", "L5VIP",
            "L6E", "L6PV", "L6SST", "L6VIP",
        ]
    )
    # Metadata handling
    parameters: List[str] = field(
        default_factory=lambda: [
            "scale_num_neurons",
            "scale_syn_strength",
            "scale_nmda_efficacy",
        ]
    )
    parameters_short: List[str] = field(
        default_factory=lambda: ["numneur", "synstr", "nmdaeff"]
    )
    # Parallelism
    n_workers: int = os.cpu_count() or 1


# Helper functions
def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _load_spike_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, times) as 1‑D arrays of dtype int64 and float64."""
    data = np.load(path, allow_pickle=False)
    return data["indices"].astype(np.int64), data["times"].astype(np.float64)


def _get_firing_rates(
    spike_dir: Path,
    populations: List[str],
    dt: float,
    t0: float,
    t1: float,
    pop_counts: Dict[str, int],
) -> np.ndarray:
    """
    Returns an array of shape (n_pop, n_bins) with firing rates (Hz) for each
    population.
    """
    T = t1 - t0
    nbins = round(T / dt)
    t = np.linspace(t0, t1, nbins + 1)
    r = np.zeros((len(populations), nbins))

    for i, pop in enumerate(populations):
        npz_path = spike_dir / f"{pop}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing spike file {npz_path}")

        _, times = _load_spike_npz(npz_path)
        counts, _ = np.histogram(times, bins=t)
        r[i, :] = counts / pop_counts[pop]            # spike count per neuron
    r /= dt                                            # convert to Hz
    return r


def _interpolate_constant(x, xp, yp, extrapolate: bool = False):
    """Piece‑wise‑constant interpolation (left‑hand convention)."""
    if extrapolate:
        yp = np.concatenate(([yp[0]], yp, [yp[-1]]))

    indices = np.searchsorted(xp, x, side="right")
    return np.array(yp)[indices]


def _get_discretized_stimulus(
    stim_group_path: Path, dt: float, t_sim_s: float
) -> np.ndarray:
    """
    Returns a 1‑D array of the stimulus value at every time bin centre.
    """
    pattern = re.compile(r"random_input_(\d+)ms\.npz")
    fnames = sorted(
        f for f in stim_group_path.iterdir() if pattern.match(f.name)
    )
    # Build limits & values vectors (seconds, not ms)
    limits = [0.0]
    values = [0.0]
    for f in fnames:
        dur_ms = int(pattern.match(f.name).group(1))
        limits.append(dur_ms / 1000.0)
        values.append(np.load(f)["arr_0"])
    limits.append(t_sim_s)

    limits = np.asarray(limits, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    T = limits[-1] - limits[0]
    nbins = round(T / dt)
    t = np.linspace(limits[0], limits[-1], nbins + 1)
    tmid = (t[1:] + t[:-1]) / 2.0
    u = _interpolate_constant(tmid, limits, values, extrapolate=True)
    return u


# HDF5 writing (executed inside each worker)
def _write_intermediate_hdf5(sim_path: Path, cfg: ConversionConfig) -> None:
    """
    Append the full simulation data to the intermediate HDF5 file.
    This function is called from inside the worker process; therefore a
    process‑wide lock (`_hdf5_lock`) is used to guarantee exclusive access.
    """
    if _intermediate_h5_path is None:
        return                     # nothing to do

    config_dict = _load_yaml(sim_path / "config.yaml")
    spike_dir = sim_path / "spike_times"
    record_path = sim_path / "record_info" / "record_neurons.data"
    metrics_path = sim_path / "metrics/avg_firing_rates.csv"
    if not os.path.isdir(sim_path) or not os.listdir(sim_path):
        return  # not a folder → ignore | empty folder → skip

    # write under lock
    with _hdf5_lock:
        with h5py.File(_intermediate_h5_path, "a") as h5:
            h5.attrs['version'] = '0.1'
            sims_grp = h5.require_group("simulations")

            sim_grp = sims_grp.create_group(sim_path.name)

            # Configuration and metadata
            for key, val in config_dict.items():
                sim_grp.attrs[key] = val

            # Spikes
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
                log_file = sim_path / log_name
                with open(log_file, "r") as fh:
                    log_text = fh.read()
                log_grp.create_dataset(log_name, data=log_text)

            # Record info
            rec_raw = np.load(record_path, allow_pickle=True)
            dtype = np.dtype([("label", h5py.string_dtype()), ("value", int)])
            arr = np.empty(len(rec_raw), dtype=dtype)
            arr["label"] = np.array(list(rec_raw.keys()))
            arr["value"] = np.array(list(rec_raw.values()))
            sim_grp.create_dataset("record_neurons", data=arr)

            # Metrics
            df = pd.read_csv(metrics_path)
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
            fnames = sorted([f for f in os.listdir(str(os.path.join(sim_path))) if pattern.match(f)])
            for fname in fnames:
                limits.append(int(pattern.match(fname).group(1)))
                values.append(np.load(os.path.join(sim_path, fname))['arr_0'])
            limits.append(config_dict['t_sim'])
            stim_grp.create_dataset("limits", data=np.array(limits))
            stim_grp.create_dataset("values", data=np.array(values))


# Core worker – processes ONE simulation folder
def _process_simulation(
    sim_path: Path,
    cfg: ConversionConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns (r (nt, n_pop), u (nt, 1), theta (n_param)).
    All arrays are already normalised (zero‑mean, unit‑var) wrt the
    running statistics supplied later by the manager.
    """
    try:
        # ----- Load configuration & parameters --------------------------------------
        config_dict = _load_yaml(sim_path / "config.yaml")
        theta = np.array([float(config_dict[p]) for p in cfg.parameters], dtype=np.float64)

        # ----- Spike‑derived firing rates -------------------------------------------
        spike_dir = sim_path / "spike_times"
        t_sim_s = float(config_dict["t_sim"]) / 1000.0

        # record_info/record_neurons.data contains a dict {pop → neuron count}
        rec_path = sim_path / "record_info" / "record_neurons.data"
        rec_raw = np.load(rec_path, allow_pickle=True)
        pop_counts = {k: int(v) for k, v in rec_raw.items()}

        firing_rate = _get_firing_rates(
            spike_dir=spike_dir,
            populations=cfg.populations,
            dt=cfg.bin_size_s,
            t0=0.0,
            t1=t_sim_s,
            pop_counts=pop_counts,
        )

        # Stimulus
        stim = _get_discretized_stimulus(sim_path, cfg.bin_size_s, t_sim_s)
        u = stim[:, None]               # make it (nt, 1)

        # Firing rate
        fr = firing_rate.T               # (nt, n_pop)

        # Optional intermediate HDF5
        if _intermediate_h5_path is not None:
            _write_intermediate_hdf5(sim_path, cfg)

        return fr, u, theta

    except Exception as exc:          # pragma: no cover – exercised in tests
        if cfg.skip_missing:
            logging.warning("Skipping %s because of %s", sim_path, exc)
            return None
        else:
            raise


# Normalisation utilities
def _normalize(x: np.ndarray, axis: Tuple[int, ...] | int):
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / sd, mu.squeeze(), sd.squeeze()


# Manager – orchestrates the parallel map, accumulates statistics, writes final file
def convert_directory(cfg: ConversionConfig) -> None:
    """Main driver – called from CLI."""
    # Initialize
    lock = ctx.Lock() if cfg.intermediate_hdf5 else None
    _worker_init(lock, str(cfg.intermediate_hdf5) if cfg.intermediate_hdf5 else None)

    logging.info("Scanning %s …", cfg.input_dir)
    sim_dirs = sorted(
        d for d in cfg.input_dir.iterdir()
        if d.is_dir() and any(d.iterdir())
    )

    if not sim_dirs:
        raise RuntimeError("No simulations found in the input directory.")

    n_sims = len(sim_dirs)
    n_pop = len(cfg.populations)
    n_params = len(cfg.parameters)

    # Intermediate HDF5 file (optional)
    # a lock is required only when the intermediate HDF5 is requested
    if cfg.intermediate_hdf5:
        # create an empty container with a version attribute
        with h5py.File(cfg.intermediate_hdf5, "w") as h:
            h.attrs["version"] = "0.1"

    # Temporary mem‑mapped files for the fast NPZ conversion
    tmp_dir = Path("_tmp_conversion")
    tmp_dir.mkdir(exist_ok=True)

    raw_r_path = tmp_dir / "r_raw.npy"
    raw_u_path = tmp_dir / "u_raw.npy"
    raw_theta_path = tmp_dir / "theta_raw.npy"

    # Process the first simulation synchronously (helps us determine nbins)
    first_fr, first_u, first_theta = _process_simulation(sim_dirs[0], cfg)  # type: ignore[assignment]
    nt = first_fr.shape[0]

    # Initialise the mem‑mapped arrays
    r_raw = np.memmap(
        raw_r_path,
        dtype=np.float64,
        mode="w+",
        shape=(n_sims, nt, n_pop),
    )
    u_raw = np.memmap(
        raw_u_path,
        dtype=np.float64,
        mode="w+",
        shape=(n_sims, nt, 1),
    )
    theta_raw = np.memmap(
        raw_theta_path,
        dtype=np.float64,
        mode="w+",
        shape=(n_sims, n_params),
    )

    # Store the first simulation (avoids an extra round‑trip to the worker)
    r_raw[0] = first_fr
    u_raw[0] = first_u
    theta_raw[0] = first_theta

    # Parallel processing of the remaining simulations
    logging.info(
        "Processing %d simulations with %d workers …",
        n_sims,
        cfg.n_workers,
    )

    # Prepare executor – note the initializer that propagates the lock & path
    with ProcessPoolExecutor(
        max_workers=cfg.n_workers,
        initializer=_worker_init,
        initargs=(lock, str(cfg.intermediate_hdf5) if cfg.intermediate_hdf5 else None),
        mp_context=ctx,
    ) as executor:
        future_to_idx = {
            executor.submit(_process_simulation, sim_path, cfg): idx
            for idx, sim_path in enumerate(sim_dirs[1:], start=1)
        }

        for future in tqdm(as_completed(future_to_idx), total=n_sims - 1):
            idx = future_to_idx[future]
            result = future.result()
            if result is None:            # skipped because of --skip-missing
                continue
            fr, u, theta = result

            # Store results in the mem‑maps
            r_raw[idx] = fr
            u_raw[idx] = u
            theta_raw[idx] = theta

    # Normalise across simulations (and time where appropriate)
    r_norm, mean_r, std_r = _normalize(r_raw, axis=(0, 1))      # (n_sims, nt, n_pop)
    u_norm, mean_u, std_u = _normalize(u_raw, axis=(0, 1))      # (n_sims, nt, 1)
    theta_norm, mean_theta, std_theta = _normalize(theta_raw, axis=0)

    # Export – NPZ (default) or single‑file HDF5
    if cfg.output_format == "npz":
        np.savez_compressed(
            cfg.output_file,
            x=r_norm,                     # (n_sims, nt, n_pop)
            u=u_norm,                     # (n_sims, nt, 1)
            theta=theta_norm,             # (n_sims, n_params)
            names_x=cfg.populations,
            names_theta=cfg.parameters_short,
            dt=cfg.bin_size_s,
            mu_x=mean_r.astype(np.float64),
            sd_x=std_r.astype(np.float64),
            mu_u=mean_u.astype(np.float64).ravel(),
            sd_u=std_u.astype(np.float64).ravel(),
            mu_theta=mean_theta.astype(np.float64),
            sd_theta=std_theta.astype(np.float64),
        )
        logging.info(
            "Saved %s (npz, %s)",
            cfg.output_file,
            "compressed" if cfg.compression == "gzip" else "uncompressed",
        )
    elif cfg.output_format == "hdf5":
        # One HDF5 file that mirrors the NPZ layout (simulations as groups)
        with h5py.File(cfg.output_file, "w", libver="latest") as h5:
            sims = h5.create_group("simulations")
            for i, sim_path in enumerate(sim_dirs):
                sim = sims.create_group(sim_path.name)
                sim.create_dataset(
                    "x",
                    data=r_raw[i],
                    compression="gzip",
                    chunks=True,
                )
                sim.create_dataset(
                    "u",
                    data=u_raw[i],
                    compression="gzip",
                    chunks=True,
                )
                sim.create_dataset(
                    "theta",
                    data=theta_raw[i],
                    compression="gzip",
                    chunks=True,
                )
                # copy the configuration attributes for convenience
                cfg_dict = _load_yaml(sim_path / "config.yaml")
                for k, v in cfg_dict.items():
                    sim.attrs[k] = v
    else:
        logging.error("Output format not supported: %s", cfg.output_format)
        raise RuntimeError(f"Output format not supported: {cfg.output_format}")

    # Clean‑up temporary mem‑mapped files
    del r_raw, u_raw, theta_raw          # close the mem‑maps
    for p in (raw_r_path, raw_u_path, raw_theta_path):
        p.unlink(missing_ok=True)
    tmp_dir.rmdir()
    logging.info("Conversion finished!")


# CLI entry‑point
def _parse_cli() -> ConversionConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Fast conversion from a workflow‑1 sweep directory to the npz / "
            "hdf5 format required by workflow‑2.  Optionally produces an "
            "intermediate HDF5 file that stores the full simulation record."
        )
    )
    parser.add_argument("-i", "--input_dir", type=Path, required=True,
                        help="Directory that contains one folder per simulation.")
    parser.add_argument("-o", "--output_file", type=Path, required=True,
                        help="Target file ( *.npz* or *.hdf5* ).")
    parser.add_argument("-h5", "--intermediate_hdf5", type=Path, default=Path("out.h5"),
                        help="Path of the intermediate HDF5 file "
                             "(stores full simulation data).")
    parser.add_argument("-b", "--bin_size", type=float, default=0.001,
                        help="Temporal bin size in seconds (default: 0.001 = 1ms).")
    parser.add_argument("-f", "--format", choices=["npz", "hdf5"], default="npz",
                        help="Output container type (default: npz).")
    parser.add_argument("-c", "--compression", default="gzip",
                        help="Compression codec (gzip for npz, blosc for zarr).")
    parser.add_argument("-l", "--compression_level", type=int, default=4,
                        help="Compression level (1‑9 for gzip, 1‑9 for blosc).")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Continue even if a simulation folder is incomplete.")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel workers (default: all CPUs).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print INFO‑level logs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    return ConversionConfig(
        input_dir=args.input_dir,
        output_file=args.output_file,
        intermediate_hdf5=args.intermediate_hdf5,
        output_format=args.format,
        compression=args.compression,
        compression_opts=args.compression_level,
        skip_missing=args.skip_missing,
        bin_size_s=args.bin_size,
        n_workers=args.workers,
    )


def main() -> None:
    cfg = _parse_cli()
    convert_directory(cfg)


if __name__ == "__main__":
    main()