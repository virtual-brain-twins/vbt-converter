# # -*- coding: utf-8 -*-
# #
# # SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# # SPDX-License-Identifier: Apache-2.0
# #

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

from converters.schema.stimuli_group import StimuliGroup


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


def get_discretized_stimulus(stim_group: StimuliGroup, dt: float) -> np.ndarray:
    """Return a discretized, piecewise‑constant stimulus trace.

    The function looks for a stimulus named ``"L1VIP_default"`` inside the
    provided ``stim_group`` and checks that it is of type
    ``"piecewise_constant"``. It then samples the stimulus at a uniform time
    step ``dt`` (seconds) and returns the values evaluated at the centre of
    each time bin.

    Parameters
    ----------
    stim_group : StimuliGroup
        Container holding a mapping from stimulus identifiers to their
        definitions. The mapping must contain a key ``"L1VIP_default"``.
    dt : float
        Desired time step for the discretized output, expressed in seconds.
        The number of bins is computed as ``round((t1 - t0) / dt)`` where
        ``t0`` and ``t1`` are the start and end times of the stimulus.

    Returns
    -------
    np.ndarray
        One‑dimensional array of stimulus values sampled at the mid‑points of
        the generated time bins.

    Raises
    ------
    ValueError
        If the expected sub‑stimulus ``"L1VIP_default"`` is not present in the
        ``stimuli_map``.
    NotImplementedError
        If the found stimulus has a ``stim_type`` other than
        ``"piecewise_constant"``.
    """
    # Obtain the stimulus applicable to any population, as it is identical for all.
    sub_stimulus = stim_group.stimuli_map.get("L1VIP_default", None)
    if sub_stimulus is None:
        raise ValueError("No sub stimulus section found")
    if sub_stimulus.stim_type != "piecewise_constant":
        raise NotImplementedError(
            f"Stimulus type '{sub_stimulus.stim_type}' not supported"
        )

    # Convert limits from ms to s for internal consistency.
    limits = np.array(sub_stimulus.limits) / 1000.0
    values = np.array(sub_stimulus.values)

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
            format produced by :class:`RateExporter`.
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
