"""Utility functions for handling AMU‐style stimulus and spike data.

This module provides a small collection of helper routines used throughout the
AMU data‑processing pipeline:

* ``interpolate_constant`` – vectorised step‑wise interpolation for piece‑wise
  constant stimulus traces.
* ``get_discretized_stimulus`` – convert a :class:`StimulusSet` section into a
  uniformly sampled 1‑D array.
* ``get_firing_rates`` – compute population‑wise firing‑rate matrices from spike
  timestamps.
* ``normalize`` – zero‑mean / unit‑std normalisation of arbitrary arrays.

All functions are written with NumPy and include explicit type hints and
Google‑style docstrings.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from converters.schema.stimulus_set import StimulusSet

def interpolate_constant(
        x: np.ndarray,
        xp: np.ndarray,
        yp: np.ndarray,
        extrapolate: bool = False,
) -> np.ndarray:
    """Vectorised step‑wise interpolation for piece‑wise constant signals.

    The function mimics a “zero‑order hold” where each interval defined by ``xp``
    takes the value of the corresponding entry in ``yp``.  If ``extrapolate`` is
    ``True`` the first and last values of ``yp`` are used to extend the signal
    beyond the outermost knots.

    Args:
        x: Points at which to evaluate the interpolated signal (shape ``(...,)``).
        xp: 1‑D array of knot positions (must be monotonically increasing).
        yp: 1‑D array of values associated with each knot; ``len(yp) == len(xp)``.
        extrapolate: If ``True`` the signal is extrapolated on both sides using
            the edge values of ``yp``; otherwise values outside ``xp`` raise an
            ``IndexError``.

    Returns:
        A NumPy array containing the interpolated values at ``x`` with the same
        shape as ``x``.

    Raises:
        IndexError: If ``extrapolate`` is ``False`` and any element of ``x`` lies
            outside the range of ``xp``.
    """
    # Extend the value vector when extrapolation is requested.
    if extrapolate:
        yp = np.concatenate(([yp[0]], yp, [yp[-1]]))

    # ``np.searchsorted`` returns the index of the right‑hand side bin for each x.
    indices = np.searchsorted(xp, x, side="right")
    return np.take(yp, indices)


def get_discretized_stimulus(stim_set: StimulusSet, dt: float) -> np.ndarray:
    """Sample a piece‑wise constant stimulus section at a uniform temporal grid.

    The stimulus is extracted from the first population of ``stim_set``.  The
    section must be of type ``"piecewise_constant"``, otherwise a
    ``NotImplementedError`` is raised.  The returned array contains the stimulus
    values evaluated at the centre of each time bin.

    Args:
        stim_set: A :class:`StimulusSet` instance containing stimulus sections.
        dt: Desired temporal resolution in seconds (must be positive).

    Returns:
        A 1‑D NumPy array of discretised stimulus values, one entry per time bin.

    Raises:
        ValueError: If the stimulus set does not contain a default section.
        NotImplementedError: If the section type is not ``"piecewise_constant"``.
    """
    stim_sect = stim_set.populations[0].sub_stimuli.get("default", None)
    if stim_sect is None:
        raise ValueError("No stimulus section found")
    if stim_sect.stim_type != "piecewise_constant":
        raise NotImplementedError(
            f"Stimulus type '{stim_sect.stim_type}' not supported"
        )

    # Convert limits from ms to s for internal consistency.
    limits = np.array(stim_sect.limits) / 1000.0
    values = np.array(stim_sect.values)

    t0, t1 = limits[0], limits[-1]
    nbins = round((t1 - t0) / dt)
    t_edges = np.linspace(t0, t1, nbins + 1)
    t_mid = (t_edges[:-1] + t_edges[1:]) / 2.0

    return interpolate_constant(t_mid, limits, values, extrapolate=True)


def get_firing_rates(
        spikes: dict[str, dict[str, np.ndarray]],
        dt: float,
        t0: float,
        t1: float,
        nneurons: dict[str, int],
        populations: list[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute population‑wise firing‑rate matrices.

    For each population listed in ``populations`` the function bins spike
    timestamps into uniform windows of width ``dt`` seconds, normalises by the
    number of recorded cells, and scales by the bin width to obtain rates in Hz.

    Args:
        spikes: Mapping from population name to a dictionary with a ``"times"``
            entry (NumPy array of spike timestamps in seconds).  This is the
            format produced by :class:`AMUExporter`.
        dt: Width of the time bins in seconds (must be positive).
        t0: Start time of the interval to analyse (seconds).
        t1: End time of the interval to analyse (seconds, ``t1 > t0``).
        nneurons: Mapping from population name to the number of recorded cells.
            Populations missing from this dictionary are assumed to contain a
            single cell.
        populations: List of population names to include in the output.

    Returns:
        A tuple ``(t_mid, rates)`` where ``t_mid`` is a 1‑D array of the bin centre
        times (shape ``(nbins,)``) and ``rates`` is a 2‑D array of shape
        ``(len(populations), nbins)`` containing firing rates (Hz) for each
        population.

    Raises:
        ValueError: If ``dt`` is non‑positive or if ``t1 <= t0``.
    """
    if dt <= 0:
        raise ValueError("dt must be a positive number")
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")

    nbins = round((t1 - t0) / dt)
    t_edges = np.linspace(t0, t1, nbins + 1)
    rates = np.zeros((len(populations), nbins), dtype=float)

    for i, pop in enumerate(populations):
        if pop not in spikes:
            continue
        times = spikes[pop]["times"]
        counts, _ = np.histogram(times, bins=t_edges)
        n_cells = nneurons.get(pop, 1)  # Guard against missing entry.
        rates[i, :] = counts / n_cells / dt

    t_mid = (t_edges[:-1] + t_edges[1:]) / 2.0
    return t_mid, rates


def normalize(
        x: np.ndarray,
        axis: tuple[int, ...] | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero‑mean / unit‑std normalisation.

    The function returns the normalised array together with the mean and standard
    deviation that were used for the transformation, enabling reversible scaling
    if required.

    Args:
        x: Input array to be normalised.
        axis: Axis or axes over which to compute the statistics.  Mirrors the
            behaviour of :func:`numpy.mean` and :func:`numpy.std`.

    Returns:
        A tuple ``(normed, mean, std)`` where:
        * ``normed`` – the zero‑mean / unit‑std version of ``x``.
        * ``mean`` – the mean of ``x`` computed over ``axis``.
        * ``std`` – the standard deviation of ``x`` computed over ``axis``.
    """
    mu = np.mean(x, axis=axis)
    sd = np.std(x, axis=axis)
    return (x - mu) / sd, mu, sd
