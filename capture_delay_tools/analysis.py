"""Analyze captured waveforms with sqe-analysis in the project environment."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from qubex.backend.quel1 import BLOCK_DURATION_NS, WORD_DURATION_NS

from ._tof import analyze as analyze_trace


@dataclass
class Waveform:
    """One captured complex waveform, with time measured in nanoseconds."""

    mux: int
    target: str
    delay: int | float
    iq: xr.DataArray
    capture_delay_word: int = 0
    delay_step_ns: float = BLOCK_DURATION_NS

    def __post_init__(self) -> None:
        """Require explicit, finite, increasing and evenly spaced time coordinates."""
        if not isinstance(self.iq, xr.DataArray):
            raise TypeError("Waveform IQ must be an xarray.DataArray.")
        if self.iq.dims != ("time_ns",) or "time_ns" not in self.iq.coords:
            raise ValueError(
                "Waveform IQ must have one dimension and an explicit time_ns coordinate."
            )
        time = self.iq.coords["time_ns"].values
        if (
            time.size < 2
            or not np.issubdtype(time.dtype, np.number)
            or np.iscomplexobj(time)
            or not np.all(np.isfinite(time))
        ):
            raise ValueError(
                "time_ns must contain at least two finite numeric coordinates."
            )
        spacing = np.diff(time)
        if not np.all(spacing > 0) or not np.allclose(spacing, spacing[0]):
            raise ValueError("time_ns must be increasing and evenly spaced.")

    @property
    def time_ns(self) -> xr.DataArray:
        """Return the nanosecond coordinate stored with the IQ values."""
        return self.iq.coords["time_ns"]

    @property
    def total_delay_ns(self) -> float:
        """Return the combined coarse and word offset in nanoseconds."""
        return (
            self.delay * self.delay_step_ns + self.capture_delay_word * WORD_DURATION_NS
        )


def analyze_waveforms(
    waveforms: list[Waveform],
    *,
    windows: dict[
        str | tuple[str, int | float] | tuple[str, int | float, int],
        tuple[float, float],
    ]
    | None = None,
    snr_threshold: float = 5.0,
    smoothing: int = 5,
) -> pd.DataFrame:
    """Locate pulse edges using sqe-analysis in the current environment."""
    if not waveforms:
        raise ValueError("Measure waveforms before analyzing them.")
    if smoothing < 1 or int(smoothing) != smoothing or snr_threshold <= 0:
        raise ValueError(
            "Smoothing must be a positive integer and SNR must be positive."
        )
    rows, metadata = [], []
    for trace in waveforms:
        iq = trace.iq
        crops = windows or {}
        window = crops.get(
            (trace.target, trace.delay, trace.capture_delay_word),
            crops.get((trace.target, trace.delay), crops.get(trace.target)),
        )
        if window is not None:
            iq = iq.sel(time_ns=slice(window[0], window[1]))
        time = iq.coords["time_ns"].values
        if len(time) < max(16, 2 * smoothing + 1) or not np.all(np.isfinite(iq.values)):
            raise ValueError(
                f"{trace.target}, delay={trace.delay}: insufficient or invalid samples."
            )
        dt = float(np.median(np.diff(time)))
        if dt <= 0 or not np.allclose(np.diff(time), dt):
            raise ValueError("Time coordinates must be increasing and evenly spaced.")
        rows.append(analyze_trace(iq, snr_threshold, int(smoothing)))
        metadata.append(
            dict(
                mux=trace.mux,
                target=trace.target,
                delay=trace.delay,
                capture_delay_word=trace.capture_delay_word,
                total_delay_ns=trace.total_delay_ns,
                time_start_ns=float(time[0]),
                time_stop_ns=float(time[-1]),
                sampling_period_ns=dt,
                pulse_duration_ns=trace.iq.attrs.get("pulse_duration_ns", np.nan),
                capture_start_ns=float(trace.time_ns[0]),
                capture_stop_ns=float(trace.time_ns[-1]),
            )
        )
    return pd.DataFrame(
        [{**meta, **row} for meta, row in zip(metadata, rows, strict=True)]
    )
