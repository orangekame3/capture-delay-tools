"""Pulse segmentation and time-of-flight fits using sqe-analysis."""

import numpy as np
import xarray as xr
from sqe_analysis.analysis import TimeOfFlightAnalysis


def analyze(iq: xr.DataArray, snr_threshold: float, smoothing: int):
    """Separate pulse edges before fitting their complex IQ transitions."""
    result: dict[str, bool | float | str | None] = dict(
        success=False,
        rise_ns=None,
        fall_ns=None,
        width_ns=None,
        snr=None,
        edge_kind="none",
        reason="low_contrast",
    )
    time = iq.coords["time_ns"].values
    values = iq.values
    envelope = np.abs(values)
    guard = max(3, smoothing)
    smooth = np.convolve(
        np.pad(envelope, (smoothing // 2, smoothing - 1 - smoothing // 2), mode="edge"),
        np.ones(smoothing) / smoothing,
        mode="valid",
    )
    # Adjacent differences estimate noise without treating pulse height as noise.
    differences = np.diff(values)
    noise = max(
        float(np.median(np.abs(differences - np.median(differences))) / np.sqrt(2)),
        1e-12,
    )
    low, high = np.percentile(smooth, [10, 90])
    contrast = float(high - low)
    if contrast < snr_threshold * noise:
        return result
    mask = smooth > low + contrast / 2
    boundaries = np.diff(np.r_[False, mask, False].astype(int))
    runs = [
        (int(a), int(b))
        for a, b in zip(
            np.flatnonzero(boundaries == 1),
            np.flatnonzero(boundaries == -1),
            strict=True,
        )
        if b - a >= guard
    ]
    if len(runs) != 1:
        result["reason"] = "multiple_pulses" if len(runs) > 1 else "no_pulse"
        return result
    start, stop = runs[0]
    rising, falling = start >= guard, stop <= len(time) - guard
    result["edge_kind"] = (
        "both"
        if rising and falling
        else "rising_only"
        if rising
        else "falling_only"
        if falling
        else "clipped"
    )
    result["reason"] = "no_edges"
    if not rising and not falling:
        return result
    midpoint = (start + stop) // 2

    def fit(begin: int, end: int):
        data = iq.isel(time_ns=slice(begin, end)).copy(deep=False)
        data.attrs = {
            **data.attrs,
            "dataset_id": data.attrs.get("dataset_id", "capture-delay"),
        }
        fitted = TimeOfFlightAnalysis().run(
            data, dim="time_ns", snr_threshold=snr_threshold, smoothing=smoothing
        )
        edge = float(fitted.params.step_location.item())
        snr = float(fitted.params.SNR.item())
        return (
            bool(fitted.success.item()) and np.isfinite(edge) and np.isfinite(snr),
            edge,
            snr,
        )

    tolerance = max(guard * float(np.median(np.diff(time))), 1e-9)
    if rising:
        ok, rise, snr = fit(0, midpoint if falling else len(time))
        if ok and abs(rise - float(time[start])) <= tolerance:
            result.update(rise_ns=rise, snr=snr)
    if falling:
        ok, fall, snr = fit(midpoint if rising else 0, len(time))
        if ok and abs(fall - float(time[stop])) <= tolerance:
            result.update(fall_ns=fall, snr=snr)
    result["success"] = result["rise_ns"] is not None or result["fall_ns"] is not None
    result["reason"] = "ok" if result["success"] else "edge_fit_failed"
    if result["rise_ns"] is not None and result["fall_ns"] is not None:
        result["width_ns"] = float(result["fall_ns"]) - float(result["rise_ns"])
    return result
