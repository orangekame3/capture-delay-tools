"""Tests for explicit export of calibrated QuEL-1 delay parameters."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

from capture_delay_tools import CaptureDelayCalibrator


def calibrator_with_results():
    """Return a calibrator with two finalized mux results."""
    calibrator = CaptureDelayCalibrator(
        cast(
            Any,
            SimpleNamespace(
                system_manager=SimpleNamespace(backend_kind="quel1"),
                config_loader=SimpleNamespace(params_path=Path.cwd()),
            ),
        ),
        readout_frequencies_ghz={0: 9.0, 2: 9.0},
    )
    calibrator.final_delays = {0: 9, 2: 8}
    calibrator.final_delay_words = {0: 4, 2: 12}
    return calibrator


def test_save_params_creates_directory_and_native_yaml(tmp_path):
    """Saving creates both native-unit YAML files without changing the final results."""
    calibrator = calibrator_with_results()
    directory = tmp_path / "new" / "params"
    paths = calibrator.save_params(directory)
    assert paths == {
        "capture_delay": directory / "capture_delay.yaml",
        "capture_delay_word": directory / "capture_delay_word.yaml",
    }
    coarse = yaml.safe_load(paths["capture_delay"].read_text())
    fine = yaml.safe_load(paths["capture_delay_word"].read_text())
    assert coarse["meta"]["unit"] == "ndelay"
    assert fine["meta"]["unit"] == "word"
    assert coarse["data"] == {0: 9, 2: 8}
    assert fine["data"] == {0: 4, 2: 12}
    assert calibrator.final_delays == {0: 9, 2: 8}
    assert calibrator.final_delay_words == {0: 4, 2: 12}


@pytest.mark.parametrize("name", ["capture_delay", "capture_delay_word"])
def test_save_params_requires_explicit_overwrite_before_any_write(tmp_path, name):
    """An existing file stops the default save without creating its partner."""
    path = tmp_path / f"{name}.yaml"
    path.write_text("existing content\n")
    with pytest.raises(FileExistsError):
        calibrator_with_results().save_params(tmp_path)
    assert path.read_text() == "existing content\n"
    assert list(tmp_path.iterdir()) == [path]


def test_save_params_merges_only_selected_muxes_and_preserves_metadata(tmp_path):
    """Explicit overwrite preserves other muxes, metadata and additional YAML fields."""
    for name, unit in [("capture_delay", "ndelay"), ("capture_delay_word", "words")]:
        (tmp_path / f"{name}.yaml").write_text(
            yaml.safe_dump(
                {
                    "meta": {
                        "unit": unit,
                        "description": "site calibration",
                        "default": 0,
                    },
                    "data": {"0": 0, 1: 16},
                    "owner": "lab",
                }
            )
        )
    paths = calibrator_with_results().save_params(str(tmp_path), overwrite=True)
    for name, expected, unit in [
        ("capture_delay", {"0": 9, 1: 16, 2: 8}, "ndelay"),
        ("capture_delay_word", {"0": 4, 1: 16, 2: 12}, "words"),
    ]:
        saved = yaml.safe_load(paths[name].read_text())
        assert saved == {
            "meta": {"unit": unit, "description": "site calibration", "default": 0},
            "data": expected,
            "owner": "lab",
        }


@pytest.mark.parametrize(
    "payload",
    [
        "not a mapping\n",
        "meta: []\ndata: {}\n",
        "meta: {unit: ns}\ndata: {}\n",
        "meta: {}\ndata: {}\n",
        "meta: {unit: word}\ndata: []\n",
        "meta: {unit: word}\ndata: {0: 0, '0': 4}\n",
        "meta: [\n",
    ],
)
def test_invalid_existing_word_yaml_does_not_change_either_file(tmp_path, payload):
    """Malformed or incompatible word YAML is rejected before touching coarse settings."""
    coarse = tmp_path / "capture_delay.yaml"
    fine = tmp_path / "capture_delay_word.yaml"
    coarse.write_text("meta: {unit: ndelay}\ndata: {0: 7}\n")
    fine.write_text(payload)
    original = coarse.read_bytes(), fine.read_bytes()
    with pytest.raises(ValueError):
        calibrator_with_results().save_params(tmp_path, overwrite=True)
    assert (coarse.read_bytes(), fine.read_bytes()) == original
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "capture_delay.yaml",
        "capture_delay_word.yaml",
    ]


@pytest.mark.parametrize(
    ("delays", "words"),
    [
        ({}, {}),
        ({0: 7}, {}),
        ({0: 7}, {1: 0}),
        ({0: -1}, {0: 0}),
        ({0: True}, {0: 0}),
        ({0: 7}, {0: 3}),
        ({0: 7}, {0: 4.0}),
        ({True: 7}, {True: 0}),
    ],
)
def test_save_params_rejects_invalid_results_before_creating_directory(
    tmp_path, delays, words
):
    """Empty or inconsistent final results never create output files."""
    calibrator = calibrator_with_results()
    calibrator.final_delays = delays
    calibrator.final_delay_words = words
    with pytest.raises(ValueError):
        calibrator.save_params(tmp_path / "params")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", [False, True])
def test_save_params_rolls_back_when_second_file_replace_fails(
    tmp_path, monkeypatch, existing
):
    """A failure replacing the word file restores the previous pair and removes temporary files."""
    import capture_delay_tools.parameter_files as files

    calibrator = calibrator_with_results()
    if existing:
        calibrator.save_params(tmp_path)
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    calibrator.final_delays = {0: 10}
    calibrator.final_delay_words = {0: 8}
    replace = files.os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        if Path(destination).name == "capture_delay_word.yaml" and not failed:
            failed = True
            raise OSError("simulated write failure")
        return replace(source, destination)

    monkeypatch.setattr(files.os, "replace", fail_once)
    with pytest.raises(OSError, match="simulated write failure"):
        calibrator.save_params(tmp_path, overwrite=True)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original


def test_export_uses_experiment_files_and_preserves_comments(tmp_path):
    """New exports retain source muxes, metadata, comments and quoted strings."""
    source = tmp_path / "source"
    source.mkdir()
    originals = {}
    for name, unit in [("capture_delay", "ndelay"), ("capture_delay_word", "word")]:
        content = (
            "# site calibration\n"
            f"meta:\n  unit: {unit}\n  description: 'our lab'\n"
            "data:\n  0: 0  # calibrated mux\n  1: 16  # untouched mux\n"
        )
        (source / f"{name}.yaml").write_text(content)
        originals[name] = content
    calibrator = calibrator_with_results()
    cast(Any, calibrator.exp).config_loader.params_path = source
    paths = calibrator.save_params(tmp_path / "export")
    for name, path in paths.items():
        text = path.read_text()
        assert "# site calibration" in text
        assert "# calibrated mux" in text
        assert "1: 16  # untouched mux" in text
        assert "description: 'our lab'" in text
        assert yaml.safe_load(text)["data"][1] == 16
        assert (source / f"{name}.yaml").read_text() == originals[name]


def test_explicit_source_and_existing_destination_precedence(tmp_path):
    """An existing destination takes precedence over an explicit source per file."""
    source = tmp_path / "source"
    destination = tmp_path / "export"
    source.mkdir()
    destination.mkdir()
    (source / "capture_delay.yaml").write_text("meta: {unit: ndelay}\ndata: {1: 7}\n")
    (source / "capture_delay_word.yaml").write_text(
        "meta: {unit: word}\ndata: {1: 12}\n"
    )
    (destination / "capture_delay.yaml").write_text(
        "meta: {unit: ndelay}\ndata: {1: 10}\n"
    )
    paths = calibrator_with_results().save_params(
        destination, source_dir=source, overwrite=True
    )
    assert yaml.safe_load(paths["capture_delay"].read_text())["data"][1] == 10
    assert yaml.safe_load(paths["capture_delay_word"].read_text())["data"][1] == 12


def test_missing_source_directory_does_not_create_output(tmp_path):
    """A missing source directory is rejected without creating destination files."""
    with pytest.raises(FileNotFoundError):
        calibrator_with_results().save_params(
            tmp_path / "output", source_dir=tmp_path / "missing"
        )
    assert not (tmp_path / "output").exists()


def test_quel3_save_params_writes_only_ns_yaml(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "capture_delay.yaml").write_text(
        "meta: {unit: ns, default: 0}\ndata: {1: 12.8}\n"
    )
    calibrator = CaptureDelayCalibrator(
        cast(
            Any,
            SimpleNamespace(
                system_manager=SimpleNamespace(backend_kind="quel3"),
                config_loader=SimpleNamespace(params_path=source),
            ),
        ),
        readout_frequencies_ghz={0: 9.0},
    )
    calibrator.final_delays = {0: 16.8}
    paths = calibrator.save_params(tmp_path / "destination")
    assert paths == {"capture_delay": tmp_path / "destination" / "capture_delay.yaml"}
    saved = yaml.safe_load(paths["capture_delay"].read_text())
    assert saved["meta"]["unit"] == "ns"
    assert saved["data"] == {0: 16.8, 1: 12.8}
    assert not (tmp_path / "destination" / "capture_delay_word.yaml").exists()
