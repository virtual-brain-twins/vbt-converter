#!/usr/bin/env python
"""
convert_dir_to_npz.py
--------------------
A fast, parallel, memory‑efficient conversion from a Neuro‑Modeling sweep
directory (workflow‑1 output) directly to the NPZ (or hdf5) format expected by
workflow‑2.

Features
~~~~~~~~
* gzip / blosc compression.
* Multi‑process (CPU‑core) parallelism.
* Detailed logging + graceful handling of missing/corrupt files.
* Simple `@dataclass` configuration – extensible for new metadata fields.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any

import h5py
import numpy as np
import yaml
from tqdm import tqdm


# Configuration dataclass
@dataclass
class ConversionConfig:
    """All knobs that a user may want to tweak."""
    # Directory / file layout
    input_dir: Path
    output_file: Path
    output_format: str = "npz"  # "npz" or "hdf5" (zarr support can be added later if needed)
    compression: str = "gzip"  # "gzip" (npz) or "blosc" (zarr)
    compression_opts: int = 4  # gzip level, or blosc clevel
    skip_missing: bool = False  # fail hard (False) or ignore missing data
    # Simulation‑wise parameters
    bin_size_s: float = 0.001  # dt in seconds (default 1 ms)
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


def _get_firing_rates(spike_dir, populations, dt, t0, t1, pop_counts):
    T = t1 - t0
    nbins = round(T / dt)
    t = np.linspace(t0, t1, nbins + 1)
    tmid = (t[1:] + t[:-1]) / 2.
    r = np.zeros((len(populations), nbins))

    for i, pop in enumerate(populations):
        npz_path = spike_dir / f"{pop}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing spike file {npz_path}")

        _, times = _load_spike_npz(npz_path)
        counts, _ = np.histogram(times, bins=t)
        r[i, :] = counts
        r[i, :] /= pop_counts[pop]
    r /= dt
    return r


def _interpolate_constant(x, xp, yp, extrapolate=False):
    if extrapolate:
        yp = np.concatenate(([yp[0]], yp, [yp[-1]]))

    indices = np.searchsorted(xp, x, side='right')
    return np.array(yp)[indices]


def _get_discretized_stimulus(stim_group_path: Path, dt, t_sim_s):
    pattern = re.compile(r"random_input_(\d+)ms\.npz")
    fnames = sorted(
        f for f in stim_group_path.iterdir()
        if pattern.match(f.name)
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
    t0 = limits[0]
    t1 = limits[-1]
    T = t1 - t0
    nbins = round(T / dt)

    # Times correspond to the bin start point.
    # This means that the first time is at t=t0. Last is not t=t1.
    t = np.linspace(t0, t1, nbins + 1)
    tmid = (t[1:] + t[:-1]) / 2.
    u = _interpolate_constant(tmid, limits, np.array(values), extrapolate=True)
    return u


# Core worker – processes ONE simulation folder
def _process_simulation(sim_path: Path,
                        cfg: ConversionConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (r (nt, n_pop), u (nt, 1), theta (n_param)).
    All arrays are **already normalised** (zero‑mean, unit‑var) wrt the
    running statistics supplied later by the manager.
    """
    try:
        # Load configuration & parameters
        config_dict = _load_yaml(sim_path / "config.yaml")
        theta = np.array([float(config_dict[p]) for p in cfg.parameters],
                         dtype=np.float64)
        spike_dir = sim_path / "spike_times"
        t_sim_s = float(config_dict["t_sim"]) / 1000.0
        rec_path = sim_path / "record_info" / "record_neurons.data"
        rec_raw = np.load(rec_path, allow_pickle=True)
        pop_counts = {k: int(v) for k, v in rec_raw.items()}
        firing_rate = _get_firing_rates(spike_dir=spike_dir,
                                           populations=cfg.populations,
                                           dt=cfg.bin_size_s,
                                           t0=0.0,
                                           t1=t_sim_s,
                                           pop_counts=pop_counts)
        stim = _get_discretized_stimulus(sim_path, cfg.bin_size_s, t_sim_s)
        u = stim[:, None]
        fr = firing_rate.T
        return fr, u, theta
    except Exception as exc:
        if cfg.skip_missing:
            logging.warning("Skipping %s because of %s", sim_path, exc)
            return None
        else:
            raise


def _normalize(x, axis):
    mu = np.mean(x, axis=axis)
    sd = np.std(x, axis=axis)
    return (x - mu) / sd, mu, sd


# Manager – orchestrates the parallel map, accumulates statistics, writes final NPZ or Zarr.
def convert_directory(cfg: ConversionConfig) -> None:
    """Main driver – called from CLI."""
    logging.info("Scanning %s …", cfg.input_dir)
    sim_dirs = sorted(d for d in cfg.input_dir.iterdir()
                      if d.is_dir() and any(d.iterdir()))

    if not sim_dirs:
        raise RuntimeError("No simulations found in the input directory.")

    n_sims = len(sim_dirs)
    n_pop = len(cfg.populations)
    n_params = len(cfg.parameters)

    # Prepare temporary mem‑mapped files (raw, un‑normalised)
    tmp_dir = Path("_tmp_conversion")
    tmp_dir.mkdir(exist_ok=True)

    raw_r_path = tmp_dir / "r_raw.npy"
    raw_u_path = tmp_dir / "u_raw.npy"
    raw_theta_path = tmp_dir / "theta_raw.npy"

    # shape: (n_sims, nt, n_pop) – we don't know nt yet, allocate after first sim.
    first_fr, first_u, first_theta = _process_simulation(sim_dirs[0], cfg)
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
    # Write the first entry (helps to keep the worker simple)
    r_raw[0] = first_fr.astype(np.float64)
    u_raw[0] = first_u.astype(np.float64)
    theta_raw[0] = first_theta.astype(np.float64)

    # Parallel processing of the remaining simulations
    logging.info("Processing %d simulations with %d workers …", n_sims, cfg.n_workers)

    with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
        # Submit jobs for all remaining sims (index 1 … n‑1)
        future_to_idx = {
            executor.submit(_process_simulation, sim_path, cfg): idx
            for idx, sim_path in enumerate(sim_dirs[1:], start=1)
        }

        for future in tqdm(as_completed(future_to_idx), total=n_sims - 1):
            idx = future_to_idx[future]
            result = future.result()
            if result is None:  # skipped because of --skip-missing
                continue
            fr, u, theta = result

            # Write to mem‑maps
            r_raw[idx] = fr.astype(np.float64)
            u_raw[idx] = u.astype(np.float64)
            theta_raw[idx] = theta.astype(np.float64)

            # Update statistics

    # Update statistics
    r_norm, mean_r, std_r = _normalize(r_raw, axis=(0, 1))  # over sims & time
    u_norm, mean_u, std_u = _normalize(u_raw, axis=(0, 1))
    theta_norm, mean_theta, std_theta = _normalize(theta_raw, axis=0)

    # Export – NPZ (default)
    if cfg.output_format == "npz":
        np.savez_compressed(
            cfg.output_file,
            x=r_norm,  # shape (n_sims, nt, n_pop)
            u=u_norm,  # shape (n_sims, nt, 1)
            theta=theta_norm,  # shape (n_sims, n_params)
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
        logging.info("Saved %s (npz, %s)", cfg.output_file,
                     "compressed" if cfg.compression == "gzip" else "uncompressed")
    elif cfg.output_format == "hdf5":
        with h5py.File(cfg.output_file, "w", libver="latest") as h5:
            sims = h5.create_group("simulations")
            for i, sim_path in enumerate(sim_dirs):
                sim = sims.create_group(sim_path.name)
                sim.create_dataset("x", data=r_raw[i], compression="gzip", chunks=True)
                sim.create_dataset("u", data=u_raw[i], compression="gzip", chunks=True)
                sim.create_dataset("theta", data=theta_raw[i], compression="gzip", chunks=True)
                # copy attributes from config.yaml
                config_dict = _load_yaml(sim_path / "config.yaml")
                for k, v in config_dict.items(): sim.attrs[k] = v
    else:
        logging.error("Output format not supported: %s", cfg.output_format)
        raise RuntimeError(f"Output format not supported: {cfg.output_format}")
    # Clean‑up temporary mem‑maps
    del r_raw, u_raw, theta_raw  # close the mem‑maps
    for p in (raw_r_path, raw_u_path, raw_theta_path):
        p.unlink(missing_ok=True)
    tmp_dir.rmdir()
    logging.info("Conversion finished!")


# CLI entry‑point
def _parse_cli() -> ConversionConfig:
    parser = argparse.ArgumentParser(
        description="Fast conversion from a workflow‑1 sweep directory to the "
                    "npz / zarr format required by workflow‑2."
    )
    parser.add_argument("-i", "--input_dir", type=Path, required=True,
                        help="Directory that contains one folder per simulation.")
    parser.add_argument("-o", "--output_file", type=Path, required=True,
                        help="Target file ( *.npz* or *.zarr* ).")
    parser.add_argument("-h5", "--intermediate_hdf5", type=Path, default=Path("out.h5"),
                        help="Path of the intermediate HDF5 file")
    parser.add_argument("-b", "--bin_size", type=float, default=0.001,
                        help="Temporal bin size in seconds (default: 0.001 = 1ms).")
    parser.add_argument("-f", "--format", choices=["npz", "zarr"], default="npz",
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
