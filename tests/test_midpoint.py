"""Pulse midpoint calibration from complete or one-sided observations."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from capture_delay_tools._tof import analyze
from capture_delay_tools.selection import finalize_delays, select_delays


@pytest.mark.parametrize("kind", ["both", "rising_only", "falling_only"])
def test_midpoint_selection_uses_duration_only_for_missing_edges(kind):
    """Either edge can locate the same absolute pulse midpoint using known duration."""
    rows = []
    for delay in [7, 8]:
        rise = 1176 - delay * 128
        fall = rise + 160
        rows.append(
            dict(
                mux=0,
                target="Q00",
                delay=delay,
                capture_delay_word=0,
                rise_ns=rise if kind != "falling_only" else np.nan,
                fall_ns=fall if kind != "rising_only" else np.nan,
                pulse_duration_ns=160,
                success=True,
                time_start_ns=0,
                time_stop_ns=510,
                sampling_period_ns=2,
            )
        )
    selected, report = select_delays(pd.DataFrame(rows), centers={0: 256})
    assert selected == {0: 7}
    assert report.selected_capture_delay_word.iloc[0] == 12
    assert report.estimated_midpoint_ns.iloc[0] == 1256


@pytest.mark.parametrize("kind", ["rising", "falling"])
def test_single_edge_fit_retains_detected_edge(kind):
    """The ToF analyzer accepts a real rising-only or falling-only waveform."""
    time = np.arange(256) * 2.0
    step = (1 + np.tanh((time - 250) / 4)) / 2
    if kind == "falling":
        step = 1 - step
    iq = xr.DataArray(
        0.01 + (1 + 0.2j) * step, dims="time_ns", coords={"time_ns": time}
    )
    result = analyze(iq, 5, 5)
    assert result["success"]
    assert result["rise_ns" if kind == "rising" else "fall_ns"] == pytest.approx(
        250, abs=4
    )
    assert result["fall_ns" if kind == "rising" else "rise_ns"] is None


def test_missing_duration_cannot_infer_single_edge():
    """One-sided data without recorded duration cannot generate a candidate."""
    table = pd.DataFrame(
        [
            dict(
                mux=0,
                target="Q00",
                delay=d,
                capture_delay_word=0,
                rise_ns=1176 - d * 128,
                fall_ns=np.nan,
                success=True,
                time_start_ns=0,
                time_stop_ns=510,
                sampling_period_ns=2,
            )
            for d in [7, 8]
        ]
    )
    selected, report = select_delays(table, centers={0: 256})
    assert not selected
    assert report.reason.iloc[0] == "insufficient_support"


def test_finalization_centers_complete_pulse_and_rejects_inferred_end():
    """Final parameters center measured bounds and never substitute an inferred edge."""
    table = pd.DataFrame(
        [
            dict(
                mux=0,
                target="Q00",
                delay=7,
                capture_delay_word=12,
                rise_ns=184.0,
                fall_ns=344.0,
                pulse_duration_ns=160,
                success=True,
                time_start_ns=0.0,
                time_stop_ns=510.0,
            )
        ]
    )
    summary = pd.DataFrame([dict(mux=0, supporting_targets=["Q00"])])
    values, words, report = finalize_delays(table, {0: 7}, summary, {0: 256})
    assert (values[0], words[0]) == (7, 12)
    assert abs(report.predicted_midpoint_ns.iloc[0] - 256) <= 16
    table.loc[0, "fall_ns"] = np.nan
    values, words, report = finalize_delays(table, {0: 7}, summary, {0: 256})
    assert not values and not words


def test_multiple_targets_center_outer_bounds_not_mean_of_midpoints():
    """Different widths are centered by the earliest rise and latest fall across targets."""
    rows = []
    for target, rise, fall in [("Q00", 1176, 1336), ("Q01", 1236, 1476)]:
        for delay in [6, 7]:
            rows.append(
                dict(
                    mux=0,
                    target=target,
                    delay=delay,
                    capture_delay_word=0,
                    rise_ns=rise - delay * 128,
                    fall_ns=fall - delay * 128,
                    pulse_duration_ns=9999,
                    success=True,
                    time_start_ns=0,
                    time_stop_ns=766,
                    sampling_period_ns=2,
                )
            )
    selected, report = select_delays(pd.DataFrame(rows), centers={0: 384})
    assert selected == {0: 7}
    assert report.selected_capture_delay_word.iloc[0] == 4
    assert report.estimated_midpoint_ns.iloc[0] == 1326


def test_inferred_long_pulse_is_not_forced_into_short_capture():
    """Known duration that exceeds the capture window rejects selection before verification."""
    rows = [
        dict(
            mux=0,
            target="Q00",
            delay=d,
            capture_delay_word=0,
            rise_ns=1176 - d * 128,
            fall_ns=np.nan,
            pulse_duration_ns=700,
            success=True,
            time_start_ns=0,
            time_stop_ns=510,
            sampling_period_ns=2,
        )
        for d in [7, 8]
    ]
    selected, report = select_delays(pd.DataFrame(rows), centers={0: 256})
    assert not selected
    assert report.reason.iloc[0] == "pulse_does_not_fit"
