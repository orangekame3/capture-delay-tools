"""Offline regression of the complete capture-delay calibration workflow."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import yaml

from capture_delay_tools import CaptureDelayCalibrator
from capture_delay_tools.analysis import analyze_waveforms


class OfflineExperiment:
    def __init__(self):
        self.calls = []
        self.readout_duration = 160.0
        self.delay_override = None
        self.frequencies = {"R00": 8.1, "R01": 8.2}
        self.qubit_labels = ["Q00", "Q01"]
        self.system_manager = SimpleNamespace(
            backend_kind="quel1",
            modified_capture_delay=self.modified_capture_delay,
            experiment_system=SimpleNamespace(
                get_mux_by_qubit=lambda q: SimpleNamespace(index=0),
                resolve_read_label=lambda q: q.replace("Q", "R"),
                control_params=SimpleNamespace(
                    get_capture_delay=lambda m: 4,
                    get_capture_delay_word=lambda m: 4,
                ),
            ),
        )

    @contextmanager
    def modified_frequencies(self, frequencies):
        original = self.frequencies.copy()
        self.frequencies.update(frequencies)
        try:
            yield
        finally:
            self.frequencies = original

    @contextmanager
    def modified_capture_delay(self, delays):
        original = self.delay_override
        self.delay_override = delays
        try:
            yield
        finally:
            self.delay_override = original

    def measure(self, sequence, readout_amplitudes=None, **kwargs):
        assert readout_amplitudes == {"R01": 0.0}
        assert all(self.frequencies[q.replace("Q", "R")] == 9.0 for q in sequence)
        return self.check_waveform(
            targets=list(sequence), capture_delay=self.delay_override, **kwargs
        )

    def connect(self):
        pass

    def configure(self):
        pass

    def check_waveform(self, targets=None, capture_delay=None, **kwargs):
        delay_ns = (capture_delay or {0: 544.0})[0]
        delay, word = divmod(int(delay_ns / 8), 16)
        self.calls.append((delay, word))
        time = np.arange(256) * 2.0
        data = {}
        for target in targets or self.qubit_labels:
            i = self.qubit_labels.index(target)
            assert self.frequencies[target.replace("Q", "R")] == 9.0
            rise = 1194.0 + i * 4 - delay * 128 - word * 8
            pulse = (np.tanh((time - rise) / 4) - np.tanh((time - rise - 160) / 4)) / 2
            rng = np.random.default_rng(123 + i)
            iq = (
                0.01
                + 0.02j
                + (1 + 0.2j) * pulse
                + 0.001 * (rng.normal(size=time.size) + 1j * rng.normal(size=time.size))
            )
            data[target] = SimpleNamespace(
                raw=iq if i == 0 else np.stack([iq, iq]), sampling_period=2.0
            )
        return SimpleNamespace(data=data)


def test_complete_calibration_centers_all_pulses_and_exports(tmp_path):
    """Coarse and word sweeps verify, center, plot and export both targets without hardware."""
    exp = OfflineExperiment()

    def make_calibrator():
        return CaptureDelayCalibrator(cast(Any, exp), readout_frequencies_ghz={0: 9.0})

    coarse = make_calibrator()
    coarse.sweep({0: range(19)}, capture_delay_word={0: [0]})
    coarse.analyze()
    coarse.select()
    assert coarse.selected_delays
    calibrator = make_calibrator()
    calibrator.sweep(
        {mux: [delay] for mux, delay in coarse.selected_delays.items()},
        capture_delay_word={0: range(0, 16, 4)},
    )
    calibrator.analyze()
    calibrator.select()
    calibrator.verify()
    report = calibrator.finalize()
    assert report.reason.tolist() == ["ready"]
    assert abs(report.predicted_midpoint_ns.iloc[0] - calibrator.centers[0]) <= 16
    figures = calibrator.check_final(show=False)
    assert figures is not None and set(figures) == {"Q00", "Q01"}
    measured = analyze_waveforms(calibrator.final_waveforms)
    assert measured.success.all()
    assert (measured.rise_ns - measured.time_start_ns >= 32).all()
    assert (measured.time_stop_ns - measured.fall_ns >= 32).all()
    for plot in [
        calibrator.plot_sweep,
        calibrator.plot_analysis,
        calibrator.plot_verification,
    ]:
        figures = plot(show=False)
        assert figures is not None and set(figures) == {"Q00", "Q01"}
    assert exp.frequencies == {"R00": 8.1, "R01": 8.2}
    assert exp.delay_override is None
    paths = calibrator.save_params(tmp_path / "saved", source_dir=tmp_path)
    assert (
        yaml.safe_load(paths["capture_delay"].read_text())["data"]
        == calibrator.final_delays
    )
    assert (
        yaml.safe_load(paths["capture_delay_word"].read_text())["data"]
        == calibrator.final_delay_words
    )
