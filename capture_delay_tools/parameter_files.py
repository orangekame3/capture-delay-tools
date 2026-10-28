"""Write QuEL-1 capture delay parameters without discarding other mux settings."""

import math
import os
import tempfile
from contextlib import ExitStack
from io import StringIO
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

from qubex.backend.quel3.capture_delay import normalize_capture_delay_ns
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from .selection import WORD_STEP


def _validate_results(delays: Mapping[int, int], words: Mapping[int, int]) -> None:
    """Require a nonempty, matching pair of valid final parameter mappings."""
    if not delays or delays.keys() != words.keys():
        raise ValueError(
            "Run finalize() first; final delays and words must contain the same muxes."
        )
    for mux in delays:
        for label, value in [
            ("MUX index", mux),
            ("capture_delay", delays[mux]),
            ("capture_delay_word", words[mux]),
        ]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(
                    f"{label} must be a non-negative integer; got {value!r}."
                )
        if words[mux] % WORD_STEP:
            raise ValueError(
                f"MUX{mux}: capture_delay_word must be a multiple of {WORD_STEP} words (32 ns)."
            )


def _merged_payload(
    path: Path, updates: Mapping[int, int | float], unit: str, codec: YAML
) -> dict[str, Any]:
    """Read one structured YAML and update only the requested mux values."""
    if path.is_symlink():
        raise ValueError(f"Cannot replace symbolic link: {path}")
    if path.exists():
        try:
            payload = codec.load(path.read_text())
        except YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}.") from exc
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("meta"), dict)
            or not isinstance(payload.get("data"), dict)
        ):
            raise ValueError(f"{path} must contain meta and data mappings.")
        configured_unit = payload["meta"].get("unit")
        allowed = {"word", "words"} if unit == "word" else {unit}
        if (
            not isinstance(configured_unit, str)
            or configured_unit.strip().lower() not in allowed
        ):
            raise ValueError(
                f"{path}: meta.unit must be {' or '.join(sorted(allowed))}."
            )
    else:
        payload = {"meta": {"unit": unit, "default": 0}, "data": {}}
    data = payload["data"]
    for mux, value in updates.items():
        mux = int(mux)
        if mux in data and str(mux) in data:
            raise ValueError(f"{path}: MUX{mux} has both integer and string keys.")
        if any(isinstance(key, bool) and key == mux for key in data):
            raise ValueError(f"{path}: MUX{mux} has a boolean key.")
        key = str(mux) if str(mux) in data else mux
        data[key] = int(value) if isinstance(value, Integral) else float(value)
    return payload


def _stage_file(directory: Path, content: bytes, cleanup: ExitStack) -> Path:
    """Stage bytes on the destination filesystem and register cleanup."""
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=".capture-delay-", delete=False
    ) as stream:
        path = Path(stream.name)
        cleanup.callback(path.unlink, missing_ok=True)
        stream.write(content)
    return path


def save_quel3_capture_delay_params(
    params_dir: str | Path,
    *,
    delays: Mapping[int, int | float],
    overwrite: bool = False,
    source_dir: str | Path,
) -> dict[str, Path]:
    """Save QuEL-3 capture delays in native nanoseconds."""
    if not delays:
        raise ValueError("Run finalize() first; final delays are empty.")
    for mux, delay in delays.items():
        if (
            isinstance(mux, bool)
            or not isinstance(mux, Integral)
            or mux < 0
            or isinstance(delay, bool)
            or not isinstance(delay, (Integral, float))
            or not math.isfinite(delay)
            or delay < 0
        ):
            raise ValueError(
                "QuEL-3 delays require non-negative integer muxes and finite ns values."
            )
        normalize_capture_delay_ns(float(delay))
    directory = Path(params_dir).expanduser()
    path = directory / "capture_delay.yaml"
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(
            f"{path} already exists; use overwrite=True to update selected muxes."
        )
    source = Path(source_dir).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Source parameter directory does not exist: {source}")
    codec = YAML()
    codec.preserve_quotes = True
    base = path if path.exists() else source / path.name
    payload = _merged_payload(base, delays, "ns", codec)
    stream = StringIO()
    codec.dump(payload, stream)
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as cleanup:
        staged = _stage_file(directory, stream.getvalue().encode(), cleanup)
        if path.exists():
            staged.chmod(path.stat().st_mode & 0o777)
        os.replace(staged, path)
    return {"capture_delay": path}


def save_capture_delay_params(
    params_dir: str | Path,
    *,
    delays: Mapping[int, int],
    words: Mapping[int, int],
    overwrite: bool = False,
    source_dir: str | Path,
) -> dict[str, Path]:
    """Validate and save both delay files, restoring earlier files if replacement fails."""
    _validate_results(delays, words)
    directory = Path(params_dir).expanduser()
    paths = {
        name: directory / f"{name}.yaml"
        for name in ["capture_delay", "capture_delay_word"]
    }
    if not overwrite:
        for path in paths.values():
            if path.exists() or path.is_symlink():
                raise FileExistsError(
                    f"{path} already exists; use overwrite=True to update selected muxes."
                )
    source = Path(source_dir).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Source parameter directory does not exist: {source}")
    codec = YAML()
    codec.preserve_quotes = True
    contents = {}
    for name, updates, unit in [
        ("capture_delay", delays, "ndelay"),
        ("capture_delay_word", words, "word"),
    ]:
        destination = paths[name]
        if destination.is_symlink():
            raise ValueError(f"Cannot replace symbolic link: {destination}")
        base = destination if destination.exists() else source / destination.name
        payload = _merged_payload(base, updates, unit, codec)
        stream = StringIO()
        codec.dump(payload, stream)
        contents[destination] = stream.getvalue().encode()
    # Validate and serialize both files before creating or replacing either.
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as cleanup:
        staged = {
            path: _stage_file(directory, content, cleanup)
            for path, content in contents.items()
        }
        backups = {
            path: _stage_file(directory, path.read_bytes(), cleanup)
            if path.exists()
            else None
            for path in contents
        }
        for path in contents:
            if path.exists():
                mode = path.stat().st_mode & 0o777
                staged[path].chmod(mode)
                backup = backups[path]
                if backup is not None:
                    backup.chmod(mode)
        replaced = []
        try:
            for path, temporary in staged.items():
                os.replace(temporary, path)
                replaced.append(path)
        except OSError:
            for path in reversed(replaced):
                backup = backups[path]
                if backup is None:
                    path.unlink()
                else:
                    os.replace(backup, path)
            raise
    return paths
