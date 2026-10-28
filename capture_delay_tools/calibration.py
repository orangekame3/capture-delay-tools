"""Local notebook workflow using Qubex measurement and visualization APIs."""

from collections.abc import Iterable
from itertools import product
from numbers import Integral, Real
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import qubex as qb
import xarray as xr
from plotly.subplots import make_subplots

from .analysis import Waveform, analyze_waveforms
from .parameter_files import save_capture_delay_params, save_quel3_capture_delay_params
from .selection import (
    RESOLUTION_NS,
    STEP_NS,
    WORD_NS,
    WORD_STEP,
    finalize_delays,
    select_delays,
)


class CaptureDelayCalibrator:
    """
    Calibrate QuEL-1 and QuEL-3 capture delays using fixed frequencies.

    Parameters
    ----------
    experiment : qb.Experiment
        Connected and configured experiment.
    readout_frequencies_ghz : dict[int, float]
        Absolute test frequency in GHz for each measured mux. Choose frequencies
        within the connected controller and signal path bands, independently of
        resonator calibration. No calibrated readout frequency is used as fallback.
    n_shots : int, optional
        Shots per target and delay point. Defaults to 256.
    snr_threshold : float, optional
        Edge detection threshold. Defaults to 5.0.
    smoothing : int, optional
        Edge analysis smoothing. Defaults to 5.

    Notes
    -----
    Each mux emits one tone from its first selected target and captures all
    selected targets simultaneously on QuEL-1. Other transmit amplitudes are zero.
    QuEL-3 checks one target at a time with readout amplitude 1.0.
    Sweep, verification and final checks use the same fixed settings.
    Frequency overrides are scoped to each mux sweep and are not exported to YAML.
    """

    def __init__(
        self,
        experiment: qb.Experiment,
        *,
        readout_frequencies_ghz: dict[int, float],
        n_shots: int = 256,
        snr_threshold: float = 5.0,
        smoothing: int = 5,
    ):
        self.backend_kind = experiment.system_manager.backend_kind
        if self.backend_kind not in {"quel1", "quel3"}:
            raise ValueError(f"Unsupported backend: {self.backend_kind!r}.")
        self._use_words = self.backend_kind == "quel1"
        self._delay_step_ns = STEP_NS if self._use_words else 1.0
        self._resolution_ns = RESOLUTION_NS
        for mux, frequency in readout_frequencies_ghz.items():
            if (
                isinstance(mux, bool)
                or not isinstance(mux, Integral)
                or mux < 0
                or isinstance(frequency, bool)
                or not isinstance(frequency, Real)
                or not np.isfinite(frequency)
                or frequency <= 0
            ):
                raise ValueError(
                    "Fixed frequencies require nonnegative integer muxes and finite positive GHz values."
                )
        self._readout_frequencies_ghz = dict(readout_frequencies_ghz)
        self.exp = experiment
        self.n_shots = n_shots
        self.snr_threshold = snr_threshold
        self.smoothing = smoothing
        self.windows = None
        self.centers = {}
        self.waveforms = []
        self.analysis = pd.DataFrame()
        self._clear_selection()

    def _clear_selection(self):
        self.selection = pd.DataFrame()
        self.selected_delays = {}
        self.verification = pd.DataFrame()
        self.verification_waveforms = []
        self._clear_final_results()

    def _clear_final_results(self) -> None:
        """Invalidate exported candidates and their final-check waveforms together."""
        self.final_delays: dict[int, int | float] = {}
        self.final_delay_words: dict[int, int] = {}
        self.final_report = pd.DataFrame()
        self.final_waveforms: list[Waveform] = []

    def _measure(
        self,
        sweep: dict[int, list[int | float]],
        *,
        capture_delay_word: dict[int, list[int]] | None = None,
    ):
        system = self.exp.system_manager.experiment_system
        available: dict[int, list[str]] = {}
        for qubit in self.exp.qubit_labels:
            index = system.get_mux_by_qubit(qubit).index
            available.setdefault(index, []).append(qubit)
        if not sweep:
            raise ValueError("Specify at least one mux and delay range.")
        if capture_delay_word is not None and capture_delay_word.keys() - sweep.keys():
            raise ValueError(
                "Word sweep muxes must also occur in the coarse delay sweep."
            )
        word_sweep = {}
        for mux, delays in sweep.items():
            if isinstance(mux, bool) or not isinstance(mux, Integral):
                raise TypeError("Sweep keys must be integer mux indices.")
            if mux not in available:
                raise ValueError(
                    f"MUX{mux}: no selected targets; available muxes: {sorted(available)}."
                )
            words = (capture_delay_word or {}).get(mux)
            if not self._use_words:
                if capture_delay_word is not None:
                    raise ValueError("capture_delay_word is not supported by QuEL-3.")
                words = [0]
            elif words is None:
                words = [system.control_params.get_capture_delay_word(mux)]
            for name, values in [
                ("capture_delay", delays),
                ("capture_delay_word", words),
            ]:
                if not values:
                    raise ValueError(f"MUX{mux}: specify at least one {name} value.")
                if any(
                    isinstance(d, bool)
                    or not isinstance(d, (Integral, Real))
                    or not np.isfinite(float(d))
                    or d < 0
                    or (name == "capture_delay_word" and not isinstance(d, Integral))
                    for d in values
                ):
                    raise ValueError(f"{name} values must be nonnegative integers.")
            if self._use_words and any(word % WORD_STEP for word in words):
                raise ValueError(
                    f"MUX{mux}: capture_delay_word must be a multiple of {WORD_STEP} words (32 ns)."
                )
            word_sweep[mux] = list(dict.fromkeys(words))
        missing = sweep.keys() - self._readout_frequencies_ghz.keys()
        if missing:
            raise ValueError(
                f"Specify readout_frequencies_ghz for muxes: {sorted(missing)}."
            )
        original = {
            mux: (
                system.control_params.get_capture_delay(mux),
                system.control_params.get_capture_delay_word(mux)
                if self._use_words
                else None,
            )
            for mux in sweep
        }
        duration = float(self.exp.readout_duration)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Readout duration must be finite and positive in ns.")
        traces = []
        try:
            for mux, delays in sweep.items():
                targets = available[mux]
                read_labels = [system.resolve_read_label(target) for target in targets]
                frequency = float(self._readout_frequencies_ghz[mux])
                # Only the first target emits a tone; all targets retain capture windows.
                amplitudes = {label: 0.0 for label in read_labels[1:]}
                sequence = {target: qb.Blank(0) for target in targets}
                with self.exp.modified_frequencies(
                    dict.fromkeys(read_labels, frequency)
                ):
                    for delay, word in product(dict.fromkeys(delays), word_sweep[mux]):
                        with self.exp.system_manager.modified_capture_delay(
                            {
                                mux: float(
                                    delay * self._delay_step_ns + int(word) * WORD_NS
                                )
                            }
                        ):
                            if self.backend_kind == "quel3":
                                results = [
                                    self.exp.check_waveform(
                                        targets=[target],
                                        readout_amplitude=1.0,
                                        readout_duration=duration,
                                        add_pump_pulses=False,
                                        n_shots=self.n_shots,
                                        plot=False,
                                    )
                                    for target in targets
                                ]
                            else:
                                result = self.exp.measure(
                                    sequence,
                                    readout_amplitudes=amplitudes,
                                    readout_duration=duration,
                                    time_integration=False,
                                    add_pump_pulses=False,
                                    n_shots=self.n_shots,
                                    plot=False,
                                )
                                results = [result]
                        for result in results:
                            for label, measured in result.data.items():
                                iq = np.asarray(measured.raw, dtype=np.complex128)
                                if iq.ndim == 2:
                                    iq = np.asarray(
                                        iq.mean(axis=0), dtype=np.complex128
                                    )
                                if iq.ndim != 1 or iq.size < 2:
                                    raise ValueError(
                                        f"{label}: expected averaged IQ with at least two samples."
                                    )
                                traces.append(
                                    Waveform(
                                        mux,
                                        label,
                                        int(delay) if self._use_words else float(delay),
                                        xr.DataArray(
                                            iq,
                                            dims=("time_ns",),
                                            coords={
                                                "time_ns": xr.DataArray(
                                                    np.arange(iq.size)
                                                    * measured.sampling_period,
                                                    dims=("time_ns",),
                                                    attrs={"units": "ns"},
                                                )
                                            },
                                            name="iq",
                                            attrs={
                                                "dataset_id": f"MUX{mux}/{label}/delay={delay}/word={word}",
                                                "readout_frequency_ghz": frequency,
                                                "pulse_duration_ns": duration,
                                            },
                                        ),
                                        capture_delay_word=int(word),
                                        delay_step_ns=self._delay_step_ns,
                                    )
                                )
        finally:
            current = {
                mux: (
                    system.control_params.get_capture_delay(mux),
                    system.control_params.get_capture_delay_word(mux)
                    if self._use_words
                    else None,
                )
                for mux in sweep
            }
            if current != original:
                raise RuntimeError("Capture-delay or word settings were not restored.")
        return traces

    def sweep(
        self,
        delays: dict[int, Iterable[int | float]],
        *,
        capture_delay_word: dict[int, Iterable[int]] | None = None,
    ) -> pd.DataFrame:
        """
        Measure coarse and word combinations sequentially for each selected mux.

        Parameters
        ----------
        delays : dict[int, Iterable[int]]
            Non-negative integer coarse offsets in 128 ns steps, keyed by mux.
        capture_delay_word : dict[int, Iterable[int]], optional
            Non-negative integer offsets in 8 ns words, divisible by 4 (32 ns).
            Each coarse range is
            combined with every specified word for the same mux. Omitted muxes
            retain their configured word value. Use one coarse value per mux
            to sweep only words. Both settings are restored after each shot batch.

        Returns
        -------
        pd.DataFrame
            Measured pairs, total offsets in ns and sample counts by target.

        Notes
        -----
        Starting a sweep invalidates previous analysis and final settings.
        """
        self._clear_selection()
        self.waveforms, self.centers = [], {}
        self.analysis = pd.DataFrame()
        self.waveforms = self._measure(
            {mux: list(values) for mux, values in delays.items()},
            capture_delay_word=(
                {mux: list(values) for mux, values in capture_delay_word.items()}
                if capture_delay_word is not None
                else None
            ),
        )
        for mux in delays:
            self.centers[mux] = float(
                np.median(
                    [
                        (
                            t.time_ns[-1].item()
                            + t.time_ns[1].item()
                            - t.time_ns[0].item()
                        )
                        / 2
                        for t in self.waveforms
                        if t.mux == mux
                    ]
                )
            )
        return pd.DataFrame(
            [
                dict(
                    mux=t.mux,
                    target=t.target,
                    delay=t.delay,
                    capture_delay_word=t.capture_delay_word,
                    total_delay_ns=t.total_delay_ns,
                    samples=len(t.iq),
                )
                for t in self.waveforms
            ]
        )

    def analyze(
        self,
        *,
        windows: dict[
            str | tuple[str, int | float] | tuple[str, int | float, int],
            tuple[float, float],
        ]
        | None = None,
    ) -> pd.DataFrame:
        """Analyze recorded traces, optionally restricting per-target time windows."""
        self._clear_selection()
        self.analysis = pd.DataFrame()
        self.windows = windows
        self.analysis = analyze_waveforms(
            self.waveforms,
            windows=windows,
            snr_threshold=self.snr_threshold,
            smoothing=self.smoothing,
        )
        return self.analysis

    def select(self) -> pd.DataFrame:
        """Select coarse and word delays centering observed or duration-inferred pulses."""
        self._clear_selection()
        if self.analysis.empty:
            raise ValueError("Run analyze() first.")
        self.selected_delays, self.selection = select_delays(
            self.analysis,
            centers=self.centers,
            delay_step_ns=self._delay_step_ns,
            resolution_ns=self._resolution_ns,
            use_words=self._use_words,
        )
        return self.selection

    def verify(self) -> pd.DataFrame:
        """Remeasure the selected pair and check the midpoint of both detected edges."""
        self.verification = pd.DataFrame()
        self.verification_waveforms = []
        self._clear_final_results()
        if not self.selected_delays:
            raise ValueError(
                "No supported candidates; inspect analysis or expand the sweep."
            )
        self.verification_waveforms = self._measure(
            {mux: [delay] for mux, delay in self.selected_delays.items()},
            capture_delay_word=(
                {
                    int(row.mux): [int(row.selected_capture_delay_word)]
                    for _, row in self.selection.iterrows()
                    if row.mux in self.selected_delays
                }
                if self._use_words
                else None
            ),
        )
        # Verification uses the complete capture, independent of sweep-specific crops.
        self.verification = analyze_waveforms(
            self.verification_waveforms,
            snr_threshold=self.snr_threshold,
            smoothing=self.smoothing,
        )
        self.verification["midpoint_ns"] = (
            self.verification.rise_ns + self.verification.fall_ns
        ) / 2
        self.verification["error_ns"] = (
            self.verification.midpoint_ns - self.verification.mux.map(self.centers)
        )
        self.verification["within_half_step"] = self.verification.success & (
            self.verification.error_ns.abs() <= self._resolution_ns / 2
        )
        return pd.DataFrame(
            self.verification[
                [
                    "mux",
                    "target",
                    "delay",
                    "capture_delay_word",
                    "rise_ns",
                    "fall_ns",
                    "midpoint_ns",
                    "error_ns",
                    "within_half_step",
                    "reason",
                ]
            ]
        )

    def finalize(self) -> pd.DataFrame:
        """Compute final overrides with a full 32 ns block before and after every pulse."""
        self._clear_final_results()
        if self.verification.empty:
            raise ValueError("Run verify() first.")
        self.final_delays, self.final_delay_words, self.final_report = finalize_delays(
            self.verification,
            self.selected_delays,
            self.selection,
            self.centers,
            delay_step_ns=self._delay_step_ns,
            resolution_ns=self._resolution_ns,
            use_words=self._use_words,
        )
        return self.final_report

    def save_params(
        self,
        params_dir: str | Path,
        *,
        overwrite: bool = False,
        source_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        """
        Save finalized delays to a QuEL-1 parameter directory.

        Parameters
        ----------
        params_dir : str | Path
            Destination directory, created if needed.
        overwrite : bool, optional
            Allow updating existing files. Other muxes and metadata are retained.
            Defaults to False, which rejects either existing delay file.

        source_dir : str | Path | None, optional
            Base parameter directory; defaults to the experiment's loaded
            `config_loader.params_path`. Existing destination files take precedence.
            Missing source files are created with native units and a zero default.

        Returns
        -------
        dict[str, Path]
            Paths keyed by `capture_delay` and `capture_delay_word`.

        Raises
        ------
        ValueError
            Final candidates are absent or invalid, or existing YAML is incompatible.
        FileExistsError
            A destination file exists and overwrite is False.
        OSError
            Writing or replacing a file fails.

        Notes
        -----
        Writes `capture_delay.yaml` in ndelay units and `capture_delay_word.yaml`
        in word units. Retains other muxes, metadata, comments and quoted strings;
        whitespace may be normalized during serialization.
        This does not change the active experiment or reload its configuration.
        Inspect `check_final()` waveforms before saving.
        """
        source = (
            self.exp.config_loader.params_path if source_dir is None else source_dir
        )
        if not self._use_words:
            return save_quel3_capture_delay_params(
                params_dir,
                delays=self.final_delays,
                overwrite=overwrite,
                source_dir=source,
            )
        return save_capture_delay_params(
            params_dir,
            delays=cast(dict[int, int], self.final_delays),
            words=self.final_delay_words,
            overwrite=overwrite,
            source_dir=source,
        )

    def check_final(self, *, show: bool = True) -> dict[str, go.Figure] | None:
        """Remeasure final overrides; return figures keyed by qubit when show=False."""
        self.final_waveforms = []
        if not self.final_delays:
            raise ValueError(
                "No verified final delays. Run finalize() after verify() and inspect "
                "calibrator.final_report; upstream operations clear both final settings."
            )
        self.final_waveforms = self._measure(
            {mux: [delay] for mux, delay in self.final_delays.items()},
            capture_delay_word=(
                {mux: [word] for mux, word in self.final_delay_words.items()}
                if self._use_words
                else None
            ),
        )
        return self.plot_final(show=show)

    def plot_final(self, *, show: bool = True) -> dict[str, go.Figure] | None:
        """Redraw stored IQ per qubit; return a qubit-to-Figure mapping when show=False."""
        return self._plot(self.final_waveforms, "Final capture", show=show)

    def _plot(
        self,
        traces: list[Waveform],
        title: str,
        *,
        heatmap: bool = False,
        table: pd.DataFrame | None = None,
        show: bool = True,
    ) -> dict[str, go.Figure] | None:
        """Build one figure per qubit and optionally display them separately."""
        if not traces:
            raise ValueError("No waveforms available for this plot.")
        figures = {
            target: self._plot_figure(
                [trace for trace in traces if trace.target == target],
                title,
                heatmap=heatmap,
                table=table,
            )
            for target in sorted({trace.target for trace in traces})
        }
        if show:
            stage = "".join(char if char.isalnum() else "_" for char in title.lower())
            for target, figure in figures.items():
                qb.viz.show_figure(figure, filename=f"capture_delay_{stage}_{target}")
            return None
        return figures

    def _plot_figure(
        self,
        traces: list[Waveform],
        title: str,
        *,
        heatmap: bool = False,
        table: pd.DataFrame | None = None,
    ) -> go.Figure:
        if not traces:
            raise ValueError("No waveforms available for this plot.")
        keys = sorted({(t.mux, t.target) for t in traces})
        fig = make_subplots(
            rows=len(keys),
            cols=1,
            subplot_titles=[
                f"MUX{mux} / {target}"
                + (
                    ""
                    if heatmap
                    else next(
                        (
                            f" / delay={t.delay}, word={t.capture_delay_word}"
                            if self._use_words
                            else f" / delay={t.delay} ns"
                        )
                        for t in traces
                        if (t.mux, t.target) == (mux, target)
                    )
                )
                for mux, target in keys
            ],
            vertical_spacing=min(0.18, 80 / (270 * len(keys))),
        )
        fig.update_annotations(font_size=15, yshift=10)
        for row, key in enumerate(keys, start=1):
            group = sorted(
                [t for t in traces if (t.mux, t.target) == key],
                key=lambda t: (t.total_delay_ns, t.delay, t.capture_delay_word),
            )
            if heatmap:
                times = group[0].time_ns.values
                if any(not np.array_equal(t.time_ns.values, times) for t in group):
                    raise ValueError(
                        "Sweep heatmaps require matching time coordinates."
                    )
                fig.add_trace(
                    go.Heatmap(
                        x=[
                            f"{t.delay}/{t.capture_delay_word}"
                            if self._use_words
                            else f"{t.delay:g}"
                            for t in group
                        ],
                        y=times,
                        z=np.asarray([np.abs(t.iq.values) for t in group]).T,
                        colorscale="Blues",
                        showscale=False,
                        hovertemplate=(
                            (
                                "delay/word %{x}<br>"
                                if self._use_words
                                else "delay %{x} ns<br>"
                            )
                            + "Time %{y} ns<br>|IQ| %{z:.3g}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                )
            else:
                trace = group[0]
                for name, values in [
                    ("I", trace.iq.real.values),
                    ("Q", trace.iq.imag.values),
                ]:
                    fig.add_trace(
                        go.Scatter(
                            x=trace.time_ns.values,
                            y=values,
                            mode="lines",
                            name=name,
                            legendgroup=name,
                            showlegend=row == 1,
                            line_color=qb.viz.COLORS[0 if name == "I" else 1],
                        ),
                        row=row,
                        col=1,
                    )
            if table is not None and not table.empty:
                subset = table[
                    (table.mux == key[0]) & (table.target == key[1]) & table.success
                ]
                for column, name, color in [
                    ("rise_ns", "Rise", qb.viz.COLORS[3]),
                    ("fall_ns", "Fall", qb.viz.COLORS[2]),
                ]:
                    for _, edge in subset.iterrows():
                        if pd.notna(cast(float, edge[column])):
                            if heatmap:
                                fig.add_trace(
                                    go.Scatter(
                                        x=[
                                            f"{int(edge.delay)}/{int(edge.capture_delay_word)}"
                                            if self._use_words
                                            else f"{float(edge.delay):g}"
                                        ],
                                        y=[edge[column]],
                                        mode="markers",
                                        marker=dict(color=color, size=7),
                                        name=name,
                                        showlegend=False,
                                    ),
                                    row=row,
                                    col=1,
                                )
                            else:
                                fig.add_vline(
                                    x=edge[column],
                                    line_color=color,
                                    line_dash="dot",
                                    row=row,  # pyright: ignore[reportArgumentType] - Plotly accepts integer indices
                                    col=1,  # pyright: ignore[reportArgumentType]
                                )
            if heatmap or table is not None:
                line_options = dict(
                    line_color=qb.viz.COLORS[5],
                    line_dash="dash",
                    annotation_text="Center",
                    annotation_font_size=12,
                    annotation_bgcolor="rgba(255,255,255,0.8)",
                )
                if heatmap:
                    fig.add_hline(
                        y=self.centers[key[0]],
                        annotation_position="top right",
                        row=row,  # pyright: ignore[reportArgumentType] - Plotly accepts integer indices
                        col=1,  # pyright: ignore[reportArgumentType]
                        exclude_empty_subplots=True,
                        **line_options,
                    )
                else:
                    fig.add_vline(
                        x=self.centers[key[0]],
                        annotation_position="top right",
                        annotation_yshift=-22,
                        row=row,  # pyright: ignore[reportArgumentType] - Plotly accepts integer indices
                        col=1,  # pyright: ignore[reportArgumentType]
                        exclude_empty_subplots=True,
                        **line_options,
                    )
            fig.update_xaxes(
                title_text=(
                    "capture_delay / capture_delay_word (128 ns / 8 ns)"
                    if self._use_words
                    else "capture_delay (ns)"
                )
                if heatmap
                else "Time (ns)",
                type="category" if heatmap else "linear",
                automargin=True,
                title_standoff=10,
                row=row,
                col=1,
            )
            fig.update_yaxes(
                title_text="Time within capture (ns)"
                if heatmap
                else "Signal (arb. units)",
                automargin=True,
                title_standoff=12,
                row=row,
                col=1,
            )
        fig.update_layout(
            template=qb.viz.DEFAULT_TEMPLATE,
            width=700,
            height=270 * len(keys) + 120,
            margin=dict(t=105, b=65, l=85, r=35),
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                y=0.99,
                yanchor="top",
                automargin=True,
            ),
            legend=dict(
                orientation="h", x=1, xanchor="right", y=1.04, yanchor="bottom"
            ),
        )
        return fig

    def plot_sweep(self, *, show: bool = True) -> dict[str, go.Figure] | None:
        """Show a heatmap per qubit; return a qubit-to-Figure mapping when show=False."""
        return self._plot(
            self.waveforms, "Capture-delay sweep |IQ|", heatmap=True, show=show
        )

    def plot_analysis(self, *, show: bool = True) -> dict[str, go.Figure] | None:
        """Mark edges per qubit; return a qubit-to-Figure mapping when show=False."""
        return self._plot(
            self.waveforms,
            "ToF: rise (red), fall (orange)",
            heatmap=True,
            table=self.analysis,
            show=show,
        )

    def plot_verification(self, *, show: bool = True) -> dict[str, go.Figure] | None:
        """Show verified IQ per qubit; return a qubit-to-Figure mapping when show=False."""
        return self._plot(
            self.verification_waveforms,
            "Centered verification",
            table=self.verification,
            show=show,
        )
