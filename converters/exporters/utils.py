from typing import Any

import numpy as np

from converters.sections.amu.stimulus_section import StimulusSection

AMU_POPULATIONS = [
    "L1VIP",
    "L23E", "L23PV", "L23SST", "L23VIP",
    "L4E", "L4PV", "L4SST", "L4VIP",
    "L5E", "L5PV", "L5SST", "L5VIP",
    "L6E", "L6PV", "L6SST", "L6VIP",
]


def interpolate_constant(x, xp, yp, extrapolate: bool = False):
    """
    Vectorised “step‑wise” interpolation used for piece‑wise constant stimuli.
    """
    if extrapolate:
        yp = np.concatenate(([yp[0]], yp, [yp[-1]]))

    indices = np.searchsorted(xp, x, side="right")
    return np.take(yp, indices)


def get_discretized_stimulus(stim_sect: StimulusSection, dt: float) -> np.ndarray:
    """
    Turn the StimulusSection into a 1‑D array sampled every ``dt`` seconds.
    """
    if stim_sect.stim_type != "piecewise_constant":
        raise NotImplementedError(f"Stimulus type '{stim_sect.stim_type}' not supported")

    limits = np.array(stim_sect.limits) / 1000.0  # ms → s
    values = np.array(stim_sect.values)

    t0, t1 = limits[0], limits[-1]
    nbins = round((t1 - t0) / dt)
    t_edges = np.linspace(t0, t1, nbins + 1)
    t_mid = (t_edges[:-1] + t_edges[1:]) / 2.0

    return interpolate_constant(t_mid, limits, values, extrapolate=True)

def get_firing_rates(
        spikes: dict,
        dt: float,
        t0: float,
        t1: float,
        nneurons: dict,
) -> Any:
    """
    Compute a (populations × time‑bins) matrix of firing rates.
    ``spikes``  – mapping ``pop → {"times": np.ndarray}`` (as built by AMUExporter)
    ``nneurons`` – mapping ``pop → n_cells`` (taken from the CSV that the original
                    pipeline stored as ``record_neurons``)
    """
    nbins = round((t1 - t0) / dt)
    t_edges = np.linspace(t0, t1, nbins + 1)
    rates = np.zeros((len(AMU_POPULATIONS), nbins), dtype=float)

    for i, pop in enumerate(AMU_POPULATIONS):
        if pop not in spikes:
            continue
        times = spikes[pop]["times"]
        counts, _ = np.histogram(times, bins=t_edges)
        n_cells = nneurons.get(pop, 1)  # guard against missing entry
        rates[i, :] = counts / n_cells / dt

    t_mid = (t_edges[:-1] + t_edges[1:]) / 2.0
    return t_mid, rates


def normalize(x: np.ndarray, axis: tuple[int, ...] | int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero‑mean / unit‑std normalisation – returns (normed, mean, std)."""
    mu = np.mean(x, axis=axis)
    sd = np.std(x, axis=axis)
    return (x - mu) / sd, mu, sd
