"""Guided wake-sample recording, resume, and export."""

import os
import tarfile

import pytest

from gva.runtime.audio import tone
from gva.runtime.wake_samples import (
    NEGATIVE_PHRASES,
    SampleSession,
    export_samples,
    holdout_split,
    negative_prompt,
    positive_prompt,
    positive_style,
    stop_voice_service,
    trim_speech,
    write_wav,
)


def test_hey_milo_style_changes_every_ten_takes():
    assert positive_style(0) == "normal"
    assert positive_style(9) == "normal"
    assert positive_style(10) == "quieter"
    assert positive_style(20) == "from across the room"
    assert positive_style(30) == "faster"
    assert positive_style(40) == "slower"
    assert positive_style(50) == "normal"
    assert positive_prompt(0) == 'Say "Hey Milo" now'
    assert "quieter" in positive_prompt(10)
    assert "across the room" in positive_prompt(29)


def test_negative_prompts_include_the_confusable_phrases_and_chat():
    shown = [negative_prompt(index) for index in range(20)]
    for phrase in ("hey Miles", "hello", "my low", "Milan", "mellow"):
        assert any(phrase in line for line in shown)
    assert "What time is it?" in NEGATIVE_PHRASES
    assert shown[5].startswith("Say ")


def test_vad_trim_drops_silence_around_a_tone():
    silence = b"\x00\x00" * 16000
    speech = tone(400, freq=220, amp=8000)
    trimmed, heard = trim_speech(silence + speech + silence, threshold=200)
    assert heard is True
    assert len(trimmed) < len(silence)
    assert len(trimmed) > len(speech) // 2
    quiet, heard_quiet = trim_speech(silence, threshold=200)
    assert heard_quiet is False
    assert quiet == silence


def test_recording_resumes_and_skips_takes_already_on_disk(tmp_path):
    session = SampleSession(tmp_path, "2026-10-26", positives=3, negatives=2, background_seconds=0.2)
    write_wav(session.positive_dir() / "take-001.wav", tone(300, amp=8000))
    seen: list[str] = []

    def capture(seconds: float) -> bytes:
        seen.append(f"{seconds:.0f}")
        return tone(300, amp=8000)

    session.record(capture, lambda: None, lambda _line: None)
    assert (session.positive_dir() / "take-001.wav").is_file()
    assert (session.positive_dir() / "take-002.wav").is_file()
    assert (session.positive_dir() / "take-003.wav").is_file()
    assert (session.negative_dir() / "take-002.wav").is_file()
    assert session.background_path().is_file()
    # take 1 was already there, so only two positive captures plus two negatives plus background.
    assert len(seen) == 5
    again: list[str] = []
    session.record(lambda _seconds: again.append("x") or b"", lambda: None, lambda _line: None)
    assert again == []
    assert "3/3 Hey Milo" in session.summary() or "3/3" in session.summary()


def test_export_archive_is_readable_and_contains_the_wavs(tmp_path):
    day = tmp_path / "samples" / "2026-10-26" / "positive"
    write_wav(day / "take-001.wav", tone(200, amp=4000))
    dest = tmp_path / "out" / "milo-samples.tar.gz"
    export_samples(tmp_path / "samples", dest, "2026-10-26")
    assert dest.is_file()
    assert oct(dest.stat().st_mode & 0o777) == "0o644"
    with tarfile.open(dest, "r:gz") as archive:
        names = archive.getnames()
    assert any(name.endswith("positive/take-001.wav") for name in names)


def test_export_refuses_an_empty_directory(tmp_path):
    with pytest.raises(SystemExit):
        export_samples(tmp_path, tmp_path / "empty.tar.gz")


def test_running_voice_service_must_be_stopped_before_recording():
    calls: list[list[str]] = []

    def run(command, capture_output=True, text=True):
        calls.append(list(command))
        if command[1] == "is-active":
            return type("R", (), {"stdout": "active\n", "returncode": 0})()
        return type("R", (), {"stdout": "denied\n", "returncode": 1})()

    with pytest.raises(SystemExit) as raised:
        stop_voice_service(run)
    assert "sudo systemctl stop gva-voice.service" in str(raised.value)
    assert calls[-1][1] == "stop"


def test_stopping_an_active_voice_service_restarts_it():
    calls: list[str] = []

    def run(command, capture_output=True, text=True):
        calls.append(command[1])
        if command[1] == "is-active":
            return type("R", (), {"stdout": "active\n", "returncode": 0})()
        return type("R", (), {"stdout": "", "returncode": 0})()

    restart = stop_voice_service(run)
    assert calls == ["is-active", "stop"]
    restart()
    assert calls[-1] == "start"


def test_inactive_voice_service_is_left_alone():
    def run(command, capture_output=True, text=True):
        return type("R", (), {"stdout": "inactive\n", "returncode": 3})()

    restart = stop_voice_service(run)
    restart()
    assert restart.__name__ == "<lambda>"


def test_holdout_keeps_a_fifth_and_leaves_small_sets_intact(tmp_path):
    paths = [tmp_path / f"take-{index:03d}.wav" for index in range(1, 11)]
    train, hold = holdout_split(paths)
    assert len(hold) == 2
    assert train + hold == paths
    assert holdout_split(paths[:4])[1] == []


def test_summary_names_the_directory(tmp_path):
    session = SampleSession(tmp_path, "2026-10-26", positives=50, negatives=20, background_seconds=180)
    write_wav(session.positive_dir() / "take-001.wav", tone(200))
    line = session.summary()
    assert "1/50 Hey Milo" in line
    assert "0/20 confusable" in line
    assert str(session.day) in line
    assert os.access(session.positive_dir() / "take-001.wav", os.R_OK)
