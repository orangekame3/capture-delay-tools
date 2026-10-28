"""Conservative selection and conversion of measured capture delays."""

import numpy as np
import pandas as pd
from qubex.backend.quel1 import BLOCK_DURATION_NS, WORD_DURATION_NS
from qubex.backend.quel1.quel1_backend_constants import (
    CAPTURE_DELAY_RESOLUTION_NS,
    CAPTURE_DELAY_WORD_STEP,
)

STEP_NS = BLOCK_DURATION_NS
WORD_NS = WORD_DURATION_NS
WORD_STEP = CAPTURE_DELAY_WORD_STEP
RESOLUTION_NS = CAPTURE_DELAY_RESOLUTION_NS


def _centered_delay(
    rises,
    falls,
    starts,
    stops,
    center,
    *,
    resolution_ns=RESOLUTION_NS,
    margin_ns=RESOLUTION_NS,
):
    """Choose the closest feasible 32 ns grid point to the envelope midpoint."""
    lower = max(0.0, float(np.max(falls - stops + margin_ns)))
    upper = float(np.min(rises - starts - margin_ns))
    lo = np.ceil(lower / resolution_ns) * resolution_ns
    hi = np.floor(upper / resolution_ns) * resolution_ns
    if hi < 0:
        return None, "insufficient_pre_margin"
    if lo > hi:
        return None, "pulse_does_not_fit"
    midpoint = (float(np.min(rises)) + float(np.max(falls))) / 2
    ideal = np.rint((midpoint - center) / resolution_ns) * resolution_ns
    return float(np.clip(ideal, lo, hi)), "ready"


def select_delays(
    table: pd.DataFrame,
    *,
    centers: dict[int, float],
    delay_step_ns: float = STEP_NS,
    resolution_ns: float = RESOLUTION_NS,
    use_words: bool = True,
) -> tuple[dict[int, int | float], pd.DataFrame]:
    """Center the pulse envelope using two consistent observations per target."""
    selected, reports = {}, []
    for mux in table["mux"].unique().tolist():
        group = table.loc[table.mux == mux]
        estimates = []
        for _, row in group.iterrows():
            if not row.success:
                continue
            rise, fall = row.get("rise_ns"), row.get("fall_ns")
            has_rise = pd.notna(rise) and np.isfinite(rise)
            has_fall = pd.notna(fall) and np.isfinite(fall)
            if not has_rise and not has_fall:
                continue
            margin = max(32, 3 * row.sampling_period_ns)
            observed = ([rise] if has_rise else []) + ([fall] if has_fall else [])
            if any(
                edge - row.time_start_ns < margin or row.time_stop_ns - edge < margin
                for edge in observed
            ):
                continue
            duration = row.get("pulse_duration_ns", np.nan)
            if not (has_rise and has_fall):
                if pd.isna(duration) or not np.isfinite(duration) or duration <= 0:
                    continue
                if not has_rise:
                    rise = fall - duration
                else:
                    fall = rise + duration
            if fall <= rise:
                continue
            offset = row.delay * delay_step_ns + row.capture_delay_word * WORD_NS
            estimates.append(
                dict(
                    target=row.target,
                    offset=offset,
                    rise=offset + rise,
                    fall=offset + fall,
                    midpoint=offset + (rise + fall) / 2,
                    start=row.get("capture_start_ns", row.time_start_ns),
                    stop=row.get("capture_stop_ns", row.time_stop_ns),
                )
            )
        supported = []
        ambiguous = False
        for target in group.target.unique():
            rows = sorted(
                [r for r in estimates if r["target"] == target],
                key=lambda r: r["midpoint"],
            )
            clusters = []
            for row in rows:
                if (
                    not clusters
                    or row["midpoint"] - clusters[-1][0]["midpoint"]
                    > max(STEP_NS, delay_step_ns, resolution_ns) / 2
                ):
                    clusters.append([])
                clusters[-1].append(row)
            valid = [c for c in clusters if len({r["offset"] for r in c}) >= 2]
            if len(valid) > 1:
                ambiguous = True
            if len(valid) == 1:
                c = valid[0]
                supported.append(
                    dict(
                        target=target,
                        points=len({r["offset"] for r in c}),
                        rise=float(np.median([r["rise"] for r in c])),
                        fall=float(np.median([r["fall"] for r in c])),
                        start=max(r["start"] for r in c),
                        stop=min(r["stop"] for r in c),
                    )
                )
        report = dict(
            mux=int(mux),
            selected_delay=None,
            selected_capture_delay_word=None,
            estimated_midpoint_ns=None,
            reason="insufficient_support",
            supporting_targets=sorted(r["target"] for r in supported),
            support_points=sum(r["points"] for r in supported),
            center_ns=centers[mux],
        )
        if ambiguous:
            report["reason"] = "conflicting_candidates"
        elif len(supported) == len(group.target.unique()):
            bounds = pd.DataFrame(supported)
            total, reason = _centered_delay(
                bounds.rise,
                bounds.fall,
                bounds.start,
                bounds.stop,
                centers[mux],
                resolution_ns=resolution_ns,
                margin_ns=RESOLUTION_NS,
            )
            report["reason"] = reason
            report["estimated_midpoint_ns"] = (
                bounds.rise.min() + bounds.fall.max()
            ) / 2
            if total is not None:
                if use_words:
                    coarse, word = divmod(
                        int(round(total / WORD_NS)), int(STEP_NS / WORD_NS)
                    )
                else:
                    coarse, word = round(float(total), 10), 0
                selected[int(mux)] = coarse
                report.update(
                    selected_delay=coarse,
                    selected_capture_delay_word=word,
                    reason="selected",
                )
        reports.append(report)
    return selected, pd.DataFrame(reports)


def finalize_delays(
    table: pd.DataFrame,
    selected: dict[int, int | float],
    summary: pd.DataFrame,
    centers: dict[int, float],
    *,
    delay_step_ns: float = STEP_NS,
    resolution_ns: float = RESOLUTION_NS,
    use_words: bool = True,
) -> tuple[dict[int, int | float], dict[int, int], pd.DataFrame]:
    """Choose an aligned delay containing every verified pulse with 32 ns margins."""
    values, words, reports = {}, {}, []
    for mux, delay in selected.items():
        group = pd.DataFrame(table.loc[(table.mux == mux) & (table.delay == delay)])
        good = pd.DataFrame(
            group.loc[
                group.success & group.rise_ns.notna() & np.isfinite(group.rise_ns)
            ]
        )
        required = set(summary.loc[summary.mux == mux, "supporting_targets"].iloc[0])
        report: dict[str, object] = dict(
            mux=mux,
            final_delay=None,
            final_capture_delay_word=None,
            reason="verification_failed",
            predicted_edge_min_ns=None,
            predicted_edge_max_ns=None,
            predicted_fall_max_ns=None,
            predicted_midpoint_ns=None,
            predicted_pre_margin_ns=None,
            predicted_post_margin_ns=None,
            unverified_targets=sorted(set(group.target) - set(good.target)),
        )
        required |= set(group.target)
        if not good.empty and required and required.issubset(set(good.target)):
            arrivals = (
                delay * delay_step_ns + good.capture_delay_word * WORD_NS + good.rise_ns
            )
            if "fall_ns" not in good or not np.isfinite(good.fall_ns).all():
                report["reason"] = "missing_falling_edge"
            elif (
                not np.isfinite(good[["rise_ns", "time_start_ns", "time_stop_ns"]])
                .all()
                .all()
                or (good.rise_ns < good.time_start_ns).any()
                or (good.fall_ns <= good.rise_ns).any()
                or (good.fall_ns > good.time_stop_ns).any()
            ):
                report["reason"] = "invalid_pulse_bounds"
            else:
                falls = (
                    delay * delay_step_ns
                    + good.capture_delay_word * WORD_NS
                    + good.fall_ns
                )
                total_ns, reason = _centered_delay(
                    arrivals,
                    falls,
                    good.time_start_ns,
                    good.time_stop_ns,
                    centers[mux],
                    resolution_ns=resolution_ns,
                    margin_ns=RESOLUTION_NS,
                )
                if total_ns is None:
                    report["reason"] = reason
                    reports.append(report)
                    continue
                predicted_rises = arrivals - total_ns
                predicted_falls = falls - total_ns
                pre_margin = float((predicted_rises - good.time_start_ns).min())
                post_margin = float((good.time_stop_ns - predicted_falls).min())
                report.update(
                    predicted_edge_min_ns=float(predicted_rises.min()),
                    predicted_edge_max_ns=float(predicted_rises.max()),
                    predicted_fall_max_ns=float(predicted_falls.max()),
                    predicted_midpoint_ns=float(
                        (predicted_rises.min() + predicted_falls.max()) / 2
                    ),
                    predicted_pre_margin_ns=pre_margin,
                    predicted_post_margin_ns=post_margin,
                )
                if total_ns < 0 or pre_margin < RESOLUTION_NS:
                    report["reason"] = "insufficient_pre_margin"
                elif post_margin < RESOLUTION_NS:
                    report["reason"] = "insufficient_post_margin"
                else:
                    if use_words:
                        coarse, word = divmod(
                            int(round(total_ns / WORD_NS)), int(STEP_NS / WORD_NS)
                        )
                    else:
                        coarse, word = round(float(total_ns), 10), 0
                    values[mux], words[mux] = coarse, word
                    report.update(
                        final_delay=coarse,
                        final_capture_delay_word=word,
                        reason="ready",
                    )
        reports.append(report)
    return values, words, pd.DataFrame(reports)
