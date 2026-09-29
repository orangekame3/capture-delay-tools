"""Tests for local coarse and word capture-delay exploration."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from capture_delay_tools.analysis import Waveform
from capture_delay_tools.calibration import CaptureDelayCalibrator
from capture_delay_tools.selection import finalize_delays, select_delays


class FakeExperiment:
    """Record temporary overrides and return deterministic offline IQ data."""

    def __init__(self):
        self.calls = []
        self.readout_duration = 160.0
        self.frequencies = {"R00": 8.1, "R01": 8.2}
        self.measured_frequencies = []
        self.frequency_contexts = 0
        self.delay_override = None
        self.qubit_labels = ["Q00"]
        self.system_manager = SimpleNamespace(
            backend_kind="quel1",
            modified_capture_delay=self.modified_capture_delay,
            experiment_system=SimpleNamespace(
                get_mux_by_qubit=lambda _: SimpleNamespace(index=0),
                resolve_read_label=lambda q: q.replace("Q", "R"),
                control_params=SimpleNamespace(
                    get_capture_delay=lambda _: 4,
                    get_capture_delay_word=lambda _: 4,
                ),
            ),
        )

    @contextmanager
    def modified_frequencies(self, frequencies):
        """Apply temporary readout frequencies and restore them on exit."""
        self.frequency_contexts += 1
        original = self.frequencies.copy()
        self.frequencies.update(frequencies)
        try:
            yield
        finally:
            self.frequencies = original

    @contextmanager
    def modified_capture_delay(self, delays):
        """Expose scoped delay overrides to the fake measurement."""
        original = self.delay_override
        self.delay_override = delays
        try:
            yield
        finally:
            self.delay_override = original

    def measure(self, sequence, **kwargs):
        """Acquire all requested targets with explicit pulse amplitudes."""
        return self.check_waveform(
            targets=list(sequence), capture_delay=self.delay_override, **kwargs
        )

    def check_waveform(self, **kwargs):
        """Return synthetic IQ and record the requested overrides."""
        self.calls.append(kwargs)
        self.measured_frequencies.append(self.frequencies.copy())
        return SimpleNamespace(
            data={
                q: SimpleNamespace(raw=np.ones(256, complex), sampling_period=2)
                for q in kwargs["targets"]
            }
        )


def test_sweep_passes_total_ns_and_records_each_pair():
    """A Cartesian sweep forwards total nanoseconds and retains every distinct word point."""
    exp = FakeExperiment()
    calibrator = CaptureDelayCalibrator(
        cast(Any, exp), readout_frequencies_ghz={0: 9.0}
    )
    result = calibrator.sweep({0: [7, 8, 7]}, capture_delay_word={0: [0, 4, 0]})
    assert [c["capture_delay"][0] for c in exp.calls] == [896, 928, 1024, 1056]
    assert all("capture_delay_word" not in c for c in exp.calls)
    assert list(zip(result.delay, result.capture_delay_word, strict=True)) == [
        (7, 0),
        (7, 4),
        (8, 0),
        (8, 4),
    ]
    assert result.total_delay_ns.tolist() == [896, 928, 1024, 1056]
    fig = calibrator.plot_sweep(show=False)
    assert fig is not None
    assert list(fig["Q00"].to_dict()["data"][0]["x"]) == ["7/0", "7/4", "8/0", "8/4"]


def test_omitted_words_use_the_configured_value():
    """Coarse-only sweeps explicitly retain and record the configured word offset."""
    exp = FakeExperiment()
    calibrator = CaptureDelayCalibrator(
        cast(Any, exp), readout_frequencies_ghz={0: 9.0}
    )
    result = calibrator.sweep({0: [7]})
    assert exp.calls[0]["capture_delay"] == {0: 928}
    assert result.capture_delay_word.tolist() == [4]


@pytest.mark.parametrize(
    "words",
    [
        {0: []},
        {1: [0]},
        {0: [-1]},
        {0: [True]},
        {0: [1.5]},
        {0: [0, 1]},
        {0: [2]},
        {0: [3]},
        {0: [5]},
    ],
)
def test_invalid_word_sweeps_do_not_measure(words):
    """Invalid word ranges fail before any hardware-backed measurement starts."""
    exp = FakeExperiment()
    calibrator = CaptureDelayCalibrator(
        cast(Any, exp), readout_frequencies_ghz={0: 9.0}
    )
    with pytest.raises((TypeError, ValueError)):
        calibrator.sweep({0: [7]}, capture_delay_word=words)
    assert exp.calls == []


def test_unknown_mux_error_reports_available_targets():
    """An unselected mux error identifies the available muxes before measurement."""
    exp = FakeExperiment()
    with pytest.raises(ValueError, match=r"available.*0"):
        CaptureDelayCalibrator(cast(Any, exp), readout_frequencies_ghz={0: 9.0}).sweep(
            {1: [7]}, capture_delay_word={1: [0]}
        )
    assert exp.calls == []


def analysis_rows():
    """Return word-only sweep rows with the same physical arrival time."""
    return pd.DataFrame(
        [
            dict(
                mux=0,
                target="Q00",
                delay=7,
                capture_delay_word=word,
                rise_ns=280 - 8 * word,
                fall_ns=440 - 8 * word,
                success=True,
                reason="ok",
                time_start_ns=0,
                time_stop_ns=510,
                sampling_period_ns=2,
            )
            for word in [0, 4]
        ]
    )


def test_selection_counts_distinct_total_delays_and_accounts_for_words():
    """Word-only sweep points support one consistent coarse centering candidate."""
    selected, report = select_delays(analysis_rows(), centers={0: 256})
    assert selected == {0: 7}
    assert report.support_points.tolist() == [2]
    assert report.reason.tolist() == ["selected"]
    table = analysis_rows()
    table.loc[1, ["delay", "capture_delay_word", "rise_ns"]] = [6, 16, 280]
    selected, report = select_delays(table, centers={0: 256})
    assert selected == {}
    assert report.reason.tolist() == ["insufficient_support"]


@pytest.mark.parametrize(
    ("rise", "expected"), [(272, (7, 12)), (280, (7, 12)), (254, (7, 8)), (288, (8, 0))]
)
def test_final_values_split_arrival_into_coarse_and_word_offsets(rise, expected):
    """Final delays center the measured pulse on the 32 ns grid with margins."""
    table = pd.DataFrame(
        [
            dict(
                mux=0,
                target="Q00",
                delay=7,
                capture_delay_word=0,
                success=True,
                rise_ns=rise,
                fall_ns=rise + 160,
                time_start_ns=0,
                time_stop_ns=510,
            )
        ]
    )
    summary = pd.DataFrame([dict(mux=0, supporting_targets=["Q00"])])
    delays, words, report = finalize_delays(table, {0: 7}, summary, {0: 256})
    assert (delays[0], words[0]) == expected
    assert abs(report.predicted_midpoint_ns.iloc[0] - 256) <= 16
    assert report.final_capture_delay_word.tolist() == [expected[1]]


def test_failed_verification_does_not_produce_final_words():
    """Unsupported verification leaves both final parameter mappings empty."""
    table = pd.DataFrame(
        [
            dict(
                mux=0,
                target="Q00",
                delay=7,
                capture_delay_word=0,
                success=False,
                rise_ns=np.nan,
            )
        ]
    )
    summary = pd.DataFrame([dict(mux=0, supporting_targets=["Q00"])])
    delays, words, report = finalize_delays(table, {0: 7}, summary, {0: 256})
    assert delays == words == {}
    assert report.reason.tolist() == ["verification_failed"]


def test_verification_and_final_measurement_use_both_overrides(monkeypatch):
    """Verification applies the selected pair and final measurement retains centered bounds."""
    exp = FakeExperiment()
    calibrator = CaptureDelayCalibrator(
        cast(Any, exp), readout_frequencies_ghz={0: 9.0}
    )
    calibrator.sweep({0: [7]}, capture_delay_word={0: [0, 4]})
    calibrator.analysis = analysis_rows()
    calibrator.select()
    monkeypatch.setattr(
        "capture_delay_tools.calibration.analyze_waveforms",
        lambda *a, **kw: (
            analysis_rows()
            .iloc[:1]
            .assign(capture_delay_word=12, rise_ns=184.0, fall_ns=344.0)
        ),
    )
    calibrator.verify()
    assert exp.calls[-1]["capture_delay"] == {0: 992}
    calibrator.finalize()
    assert calibrator.final_delays == {0: 7}
    assert calibrator.final_delay_words == {0: 12}
    fig = calibrator.check_final(show=False)
    assert fig is not None
    assert "word=12" in fig["Q00"].layout.annotations[0].text
    calls = len(exp.calls)
    redrawn = calibrator.plot_final(show=False)
    assert redrawn is not None
    assert redrawn["Q00"].to_json() == fig["Q00"].to_json()
    assert len(exp.calls) == calls
    assert exp.calls[-1]["capture_delay"] == {0: 992}
    assert "capture_delay_word" not in exp.calls[-1]
    calibrator.select()
    assert calibrator.final_delays == calibrator.final_delay_words == {}
    with pytest.raises(ValueError, match="finalize"):
        calibrator.check_final(show=False)
    with pytest.raises(ValueError, match="No waveforms"):
        calibrator.plot_final(show=False)


def test_analysis_preserves_words_and_prefers_pair_specific_windows(monkeypatch):
    """Analysis metadata and crop selection distinguish equal coarse delays with different words."""
    from capture_delay_tools.analysis import analyze_waveforms

    starts = []

    def analyze(iq, snr_threshold, smoothing):
        assert isinstance(iq, xr.DataArray)
        starts.append(iq.time_ns[0].item())
        return dict(success=True)

    monkeypatch.setattr("capture_delay_tools.analysis.analyze_trace", analyze)
    traces = [
        Waveform(
            0,
            "Q00",
            7,
            xr.DataArray(
                np.ones(64, dtype=np.complex128),
                dims="time_ns",
                coords={"time_ns": np.arange(64) * 2},
            ),
            capture_delay_word=w,
        )
        for w in [0, 4]
    ]
    table = analyze_waveforms(traces, windows={("Q00", 7, 4): (20, 100)})
    assert starts == [0, 20]
    assert table.capture_delay_word.tolist() == [0, 4]
    assert table.total_delay_ns.tolist() == [896, 928]


def test_omitted_unaligned_words_fail_before_measurement():
    """A configured unaligned offset fails before starting a coarse sweep."""
    exp = FakeExperiment()
    exp.system_manager.experiment_system.control_params.get_capture_delay_word = (
        lambda _: 1
    )
    with pytest.raises(ValueError, match="multiple of 4"):
        CaptureDelayCalibrator(cast(Any, exp), readout_frequencies_ghz={0: 9.0}).sweep(
            {0: [7]}
        )
    assert exp.calls == []


def test_local_sqe_analysis_fits_synthetic_pulse():
    """The project environment fits a pulse without a subprocess or hardware."""
    from capture_delay_tools.analysis import analyze_waveforms

    time = np.arange(256) * 2.0
    pulse = (np.tanh((time - 120) / 4) - np.tanh((time - 360) / 4)) / 2
    rng = np.random.default_rng(17)
    iq = (
        0.01
        + 0.02j
        + (1 + 0.2j) * pulse
        + 0.001 * (rng.normal(size=time.size) + 1j * rng.normal(size=time.size))
    )
    result = analyze_waveforms(
        [
            Waveform(
                0, "Q00", 7, xr.DataArray(iq, dims="time_ns", coords={"time_ns": time})
            )
        ]
    )
    assert result.success.iloc[0]
    assert result.rise_ns.iloc[0] == pytest.approx(120, abs=4)
    assert result.fall_ns.iloc[0] == pytest.approx(360, abs=4)


def test_sweep_keeps_iq_and_time_coordinates_together():
    """Measured IQ retains its nanosecond coordinate and shot average in a DataArray."""
    exp = FakeExperiment()
    calibrator = CaptureDelayCalibrator(
        cast(Any, exp), readout_frequencies_ghz={0: 9.0}
    )
    report = calibrator.sweep({0: [7]}, capture_delay_word={0: [4]})
    trace = calibrator.waveforms[0]
    assert isinstance(report, pd.DataFrame)
    assert isinstance(trace.iq, xr.DataArray)
    assert trace.iq.dims == ("time_ns",)
    np.testing.assert_array_equal(trace.iq.time_ns.values, np.arange(256) * 2)
    np.testing.assert_array_equal(trace.iq.values, np.ones(256, complex))
    assert trace.iq.time_ns.attrs["units"] == "ns"
    assert trace.iq.attrs["pulse_duration_ns"] == exp.readout_duration
    assert exp.calls[0]["readout_duration"] == exp.readout_duration


@pytest.mark.parametrize("times", [[0, 2, 2], [4, 2, 0], [0, 2, 5], [0, 2, np.nan]])
def test_waveform_rejects_invalid_time_coordinates(times):
    """Invalid time coordinates fail before coordinate slicing can hide the problem."""
    iq = xr.DataArray(np.ones(3, complex), dims="time_ns", coords={"time_ns": times})
    with pytest.raises(ValueError, match="time_ns"):
        Waveform(0, "Q00", 7, iq)


@pytest.mark.parametrize(
    "iq",
    [
        xr.DataArray(np.ones(32, complex), dims="sample"),
        xr.DataArray(np.ones(32, complex), dims="time_ns"),
        xr.DataArray(
            np.ones((2, 32), complex),
            dims=("shot", "time_ns"),
            coords={"time_ns": np.arange(32)},
        ),
    ],
)
def test_waveform_requires_explicit_one_dimensional_time_coordinate(iq):
    """Waveforms require a one-dimensional IQ array with an explicit time_ns coordinate."""
    with pytest.raises(ValueError, match="time_ns"):
        Waveform(0, "Q00", 7, iq)


def test_analysis_crops_different_length_waveforms_without_changing_originals(
    monkeypatch,
):
    """Coordinate crops preserve attributes and support independently sized waveforms."""
    from capture_delay_tools.analysis import analyze_waveforms

    originals = [
        xr.DataArray(
            np.ones(n, complex),
            dims="time_ns",
            coords={"time_ns": np.arange(n) * 2.0},
            attrs={"dataset_id": f"capture-{n}"},
        )
        for n in [64, 96]
    ]
    traces = [Waveform(0, "Q00", 7, originals[0]), Waveform(0, "Q01", 7, originals[1])]
    captured = []

    def analyze(iq, snr_threshold, smoothing):
        captured.append(iq)
        return {"success": True}

    monkeypatch.setattr("capture_delay_tools.analysis.analyze_trace", analyze)
    report = analyze_waveforms(traces, windows={"Q00": (20, 100)})
    xr.testing.assert_identical(captured[0], originals[0].sel(time_ns=slice(20, 100)))
    xr.testing.assert_identical(captured[1], originals[1])
    assert report.time_start_ns.tolist() == [20, 0]
    assert report.time_stop_ns.tolist() == [100, 190]
    assert originals[0].sizes["time_ns"] == 64


def test_single_tone_acquires_all_targets_once_per_delay_and_restores():
    """A mux sweep sends one tone and captures all targets with one frequency context."""
    exp = FakeExperiment()
    exp.qubit_labels = ["Q00", "Q01"]
    original = exp.frequencies.copy()
    calibrator = CaptureDelayCalibrator(
        cast(Any, exp), readout_frequencies_ghz={0: 11.0}
    )
    calibrator.sweep({0: [7, 8]})
    assert len(exp.calls) == 2
    assert all(c["targets"] == ["Q00", "Q01"] for c in exp.calls)
    assert all(c["readout_amplitudes"] == {"R01": 0.0} for c in exp.calls)
    assert all(c["time_integration"] is False for c in exp.calls)
    assert exp.measured_frequencies == [{"R00": 11.0, "R01": 11.0}] * 2
    assert exp.frequency_contexts == 1
    assert exp.frequencies == original
    assert exp.delay_override is None
    assert len(calibrator.waveforms) == 4
    assert all(
        t.iq.attrs["readout_frequency_ghz"] == 11.0 for t in calibrator.waveforms
    )


def test_measurement_failure_restores_fixed_frequency(monkeypatch):
    """A waveform acquisition failure exits the frequency override context."""
    exp = FakeExperiment()
    original = exp.frequencies.copy()

    def fail(**kwargs):
        assert exp.frequencies["R00"] == 9.0
        raise RuntimeError("acquisition failed")

    monkeypatch.setattr(exp, "check_waveform", fail)
    with pytest.raises(RuntimeError, match="acquisition failed"):
        CaptureDelayCalibrator(cast(Any, exp), readout_frequencies_ghz={0: 9.0}).sweep(
            {0: [7]}
        )
    assert exp.frequencies == original


@pytest.mark.parametrize(
    "frequencies",
    [
        {},
        {1: 9.0},
        {0: float("nan")},
        {0: float("inf")},
        {0: -1},
        {0: True},
        {True: 9.0},
        {0: "9.0"},
    ],
)
def test_invalid_or_missing_fixed_frequency_does_not_measure(frequencies):
    """Invalid or missing fixed frequencies fail before any acquisition."""
    exp = FakeExperiment()
    with pytest.raises(ValueError):
        CaptureDelayCalibrator(
            cast(Any, exp), readout_frequencies_ghz=frequencies
        ).sweep({0: [7]})
    assert exp.calls == []


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"fall_ns": np.nan}, "missing_falling_edge"),
        ({"fall_ns": 600.0}, "invalid_pulse_bounds"),
        ({"fall_ns": 270.0}, "invalid_pulse_bounds"),
        ({"rise_ns": np.inf}, "verification_failed"),
        (
            {"fall_ns": 500.0, "time_stop_ns": 510.0, "time_start_ns": 250.0},
            "pulse_does_not_fit",
        ),
    ],
)
def test_final_candidates_require_complete_pulse_and_both_margins(changes, reason):
    """Incomplete pulses or insufficient final trailing space cannot produce saved delays."""
    table = analysis_rows().iloc[:1].copy()
    table = table.astype({"rise_ns": float, "fall_ns": float})
    for key, value in changes.items():
        table.loc[0, key] = value
    summary = pd.DataFrame([dict(mux=0, supporting_targets=["Q00"])])
    delays, words, report = finalize_delays(table, {0: 7}, summary, {0: 256})
    assert delays == words == {}
    assert report.reason.tolist() == [reason]


def test_final_candidate_checks_targets_without_selection_support():
    """A failing measured target prevents approval even when it did not support selection."""
    table = analysis_rows().iloc[:1].copy()
    other = table.copy()
    other["target"] = "Q01"
    other["success"] = False
    table = pd.concat([table, other], ignore_index=True)
    summary = pd.DataFrame([dict(mux=0, supporting_targets=["Q00"])])
    delays, words, report = finalize_delays(table, {0: 7}, summary, {0: 256})
    assert delays == words == {}
    assert report.unverified_targets.iloc[0] == ["Q01"]


def test_final_candidate_rejects_negative_delay_needed_for_margin():
    """An early pulse that cannot have a full leading block is rejected without clamping."""
    table = analysis_rows().iloc[:1].copy()
    table.loc[0, ["delay", "rise_ns", "fall_ns"]] = [0, 16, 160]
    summary = pd.DataFrame([dict(mux=0, supporting_targets=["Q00"])])
    delays, words, report = finalize_delays(table, {0: 0}, summary, {0: 16})
    assert delays == words == {}
    assert report.reason.tolist() == ["insufficient_pre_margin"]


def test_final_report_includes_trailing_edge_and_minimum_margins():
    """Final reports expose whole-pulse containment and accept exactly 32 ns of trailing space."""
    table = analysis_rows().iloc[:1].copy()
    table.loc[0, "time_start_ns"] = 224.0
    table.loc[0, "fall_ns"] = 478.0
    summary = pd.DataFrame([dict(mux=0, supporting_targets=["Q00"])])
    delays, words, report = finalize_delays(table, {0: 7}, summary, {0: 256})
    assert delays == {0: 7} and words == {0: 0}
    assert report.predicted_pre_margin_ns.iloc[0] == 56
    assert report.predicted_fall_max_ns.iloc[0] == 478
    assert report.predicted_post_margin_ns.iloc[0] == 32


@pytest.mark.parametrize(
    "method", ["plot_sweep", "plot_analysis", "plot_verification", "plot_final"]
)
def test_plots_return_one_independent_figure_per_qubit(method, monkeypatch):
    """All plot stages return separate qubit figures and display each with a distinct filename."""
    exp = FakeExperiment()
    exp.qubit_labels = ["Q00", "Q01"]
    calibrator = CaptureDelayCalibrator(
        cast(Any, exp), readout_frequencies_ghz={0: 11.0}
    )
    calibrator.sweep({0: [7]})
    calibrator.verification_waveforms = calibrator.waveforms
    calibrator.final_waveforms = calibrator.waveforms
    calls = len(exp.calls)
    plot = getattr(calibrator, method)
    figures = plot(show=False)
    assert set(figures) == {"Q00", "Q01"}
    for target, fig in figures.items():
        assert target in fig.layout.annotations[0].text
        assert len(fig.data) == (1 if method in {"plot_sweep", "plot_analysis"} else 2)
        assert "yaxis2" not in fig.layout
    displayed = []
    monkeypatch.setattr(
        "capture_delay_tools.calibration.qb.viz.show_figure",
        lambda fig, **kw: displayed.append(kw["filename"]),
    )
    assert plot() is None
    assert len(set(displayed)) == 2
    assert all(any(q in filename for filename in displayed) for q in figures)
    assert len(exp.calls) == calls
