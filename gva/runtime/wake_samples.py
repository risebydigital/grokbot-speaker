"""Record and pack real wake-word clips from the AIRHUG.

`gva wake record` writes 16 kHz mono WAVs under
``/var/lib/gva/wake-samples/<date>/{positive,negative,background}``.
A take that is already on disk is skipped, so stopping halfway and
running the command again continues the same day.
"""

from __future__ import annotations

import os
import pwd
import subprocess
import tarfile
import wave
from collections.abc import Callable
from pathlib import Path

from gva.runtime.audio import rms

SAMPLES_ROOT = Path("/var/lib/gva/wake-samples")
VOICE_UNIT = "gva-voice.service"
EXPORT_PATH = Path("/tmp/milo-samples.tar.gz")
SAMPLE_RATE = 16000

# One style for each block of ten Hey Milo takes, then the list repeats.
POSITIVE_STYLES = (
    "normal",
    "quieter",
    "from across the room",
    "faster",
    "slower",
)

# Twenty takes by default: the confusable phrases, then ordinary sentences, then the phrases again.
NEGATIVE_PHRASES = (
    "hey Miles",
    "hello",
    "my low",
    "Milan",
    "mellow",
    "What time is it?",
    "Can you put the kettle on?",
    "The weather looks fine today.",
    "I'll be back in a minute.",
    "Did you feed the cat?",
    "hey Miles",
    "hello",
    "my low",
    "Milan",
    "mellow",
    "Turn the lamp off please.",
    "How long is the timer?",
    "I left my keys in the kitchen.",
    "See you after lunch.",
    "That was a long day.",
)


def positive_style(index: int) -> str:
    """Style for a zero-based Hey Milo take. It changes every ten takes."""
    return POSITIVE_STYLES[(index // 10) % len(POSITIVE_STYLES)]


def positive_prompt(index: int) -> str:
    style = positive_style(index)
    if style == "normal":
        return 'Say "Hey Milo" now'
    return f'Say "Hey Milo" now, {style}'


def negative_prompt(index: int) -> str:
    phrase = NEGATIVE_PHRASES[index % len(NEGATIVE_PHRASES)]
    return f'Say "{phrase}" now'


def listen_seconds(prompt: str) -> float:
    """Short phrases get four seconds. A sentence gets six."""
    phrase = prompt.split('"')[1] if '"' in prompt else prompt
    if len(phrase.split()) >= 4:
        return 6.0
    return 4.0


def trim_speech(
    pcm: bytes,
    sample_rate: int = SAMPLE_RATE,
    chunk_ms: int = 20,
    threshold: float = 500,
    pad_ms: int = 200,
    min_speech_ms: int = 200,
) -> tuple[bytes, bool]:
    """Keep the speech, plus a little silence on each side.

    Returns ``(audio, heard)``. When nothing clears the energy threshold the
    original buffer is returned and ``heard`` is false.
    """
    frame_bytes = int(sample_rate * chunk_ms / 1000) * 2
    if frame_bytes <= 0 or len(pcm) < frame_bytes:
        return pcm, False
    flags: list[bool] = []
    for start in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        flags.append(rms(pcm[start : start + frame_bytes]) >= threshold)
    if not any(flags):
        return pcm, False
    first = next(index for index, speech in enumerate(flags) if speech)
    last = len(flags) - 1 - next(index for index, speech in enumerate(reversed(flags)) if speech)
    speech_ms = (last - first + 1) * chunk_ms
    if speech_ms < min_speech_ms:
        return pcm, False
    pad_frames = max(0, int(pad_ms / chunk_ms))
    start = max(0, first - pad_frames) * frame_bytes
    end = min(len(pcm), (last + 1 + pad_frames) * frame_bytes)
    return pcm[start:end], True


def write_wav(path: Path, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a mono WAV, replacing the destination only after the file is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    os.chmod(temporary, 0o644)
    temporary.replace(path)


def wav_seconds(path: Path, sample_rate: int = SAMPLE_RATE) -> float:
    if not path.is_file():
        return 0.0
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate() or sample_rate)


def take_path(directory: Path, index: int) -> Path:
    return directory / f"take-{index:03d}.wav"


class SampleSession:
    """One day's clips. Existing take files are left in place."""

    def __init__(
        self,
        root: Path,
        date: str,
        positives: int = 50,
        negatives: int = 20,
        background_seconds: float = 180,
        energy_threshold: float = 500,
    ):
        self.root = Path(root)
        self.date = date
        self.day = self.root / date
        self.positives = positives
        self.negatives = negatives
        self.background_seconds = background_seconds
        self.energy_threshold = energy_threshold

    def positive_dir(self) -> Path:
        return self.day / "positive"

    def negative_dir(self) -> Path:
        return self.day / "negative"

    def background_path(self) -> Path:
        return self.day / "background" / "room.wav"

    def missing_takes(self, directory: Path, count: int) -> list[int]:
        return [index for index in range(1, count + 1) if not take_path(directory, index).is_file()]

    def background_needed(self) -> bool:
        return wav_seconds(self.background_path()) < self.background_seconds * 0.9

    def complete(self) -> bool:
        return (
            not self.missing_takes(self.positive_dir(), self.positives)
            and not self.missing_takes(self.negative_dir(), self.negatives)
            and not self.background_needed()
        )

    def summary(self) -> str:
        positives = len(list(self.positive_dir().glob("take-*.wav"))) if self.positive_dir().is_dir() else 0
        negatives = len(list(self.negative_dir().glob("take-*.wav"))) if self.negative_dir().is_dir() else 0
        background = wav_seconds(self.background_path())
        return (
            f"Saved {positives}/{self.positives} Hey Milo, {negatives}/{self.negatives} confusable, "
            f"and {background:.0f}s of room background in {self.day}."
        )

    def record(
        self,
        capture: Callable[[float], bytes],
        beep: Callable[[], None],
        say: Callable[[str], None],
    ) -> str:
        """Record whatever is still missing. ``capture`` returns 16 kHz mono PCM."""
        self.day.mkdir(parents=True, exist_ok=True)
        os.chmod(self.day, 0o755)
        self._record_takes(
            self.positive_dir(),
            self.missing_takes(self.positive_dir(), self.positives),
            self.positives,
            positive_prompt,
            capture,
            beep,
            say,
            trim=True,
        )
        self._record_takes(
            self.negative_dir(),
            self.missing_takes(self.negative_dir(), self.negatives),
            self.negatives,
            negative_prompt,
            capture,
            beep,
            say,
            trim=True,
        )
        if self.background_needed():
            say(f"Stay quiet. Recording the room for {self.background_seconds / 60:.0f} minutes.")
            beep()
            pcm = capture(self.background_seconds)
            write_wav(self.background_path(), pcm)
            say(f"Saved {self.background_path()}")
        say(f"Files: {self.day}")
        line = self.summary()
        say(line)
        return line

    def _record_takes(
        self,
        directory: Path,
        pending: list[int],
        total: int,
        prompt_for: Callable[[int], str],
        capture: Callable[[float], bytes],
        beep: Callable[[], None],
        say: Callable[[str], None],
        trim: bool,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o755)
        for index in pending:
            prompt = prompt_for(index - 1)
            say(f"{index}/{total}  {prompt}")
            pcm, heard = self._capture_take(capture, beep, listen_seconds(prompt), trim)
            if not heard:
                say("Didn't hear that. One more try.")
                pcm, heard = self._capture_take(capture, beep, listen_seconds(prompt), trim)
            if not heard:
                say("Still quiet. Saving this take as it was recorded.")
            path = take_path(directory, index)
            write_wav(path, pcm)
            say(f"Saved {path.name}")

    def _capture_take(
        self,
        capture: Callable[[float], bytes],
        beep: Callable[[], None],
        seconds: float,
        trim: bool,
    ) -> tuple[bytes, bool]:
        beep()
        pcm = capture(seconds)
        if not trim:
            return pcm, True
        trimmed, heard = trim_speech(pcm, threshold=self.energy_threshold)
        return (trimmed if heard else pcm), heard


def stop_voice_service(run: Callable[..., subprocess.CompletedProcess] | None = None) -> Callable[[], None]:
    """Stop ``gva-voice`` when it is active. The returned function starts it again.

    When the service is running and ``systemctl stop`` fails, this raises
    ``SystemExit`` with the command to run. Recording does not start while
    the voice service still holds the microphone.
    """
    spawn = run or subprocess.run
    try:
        active = spawn(["systemctl", "is-active", VOICE_UNIT], capture_output=True, text=True)
    except FileNotFoundError:
        return lambda: None
    if active.stdout.strip() != "active":
        return lambda: None
    stopped = spawn(["systemctl", "stop", VOICE_UNIT], capture_output=True, text=True)
    if stopped.returncode != 0:
        raise SystemExit(
            "gva-voice.service is using the microphone and could not be stopped.\n"
            "Run:\n"
            "  sudo systemctl stop gva-voice.service\n"
            "  sudo gva wake record\n"
            "The record command starts gva-voice again when it finishes."
        )

    def restart() -> None:
        spawn(["systemctl", "start", VOICE_UNIT], capture_output=True, text=True)

    return restart


def holdout_split(paths: list[Path], fraction: float = 0.2) -> tuple[list[Path], list[Path]]:
    """Last slice is held out. Fewer than five clips all stay in training."""
    ordered = sorted(paths)
    if len(ordered) < 5:
        return ordered, []
    hold = max(1, int(round(len(ordered) * fraction)))
    hold = min(hold, len(ordered) - 1)
    return ordered[:-hold], ordered[-hold:]


def export_samples(root: Path, dest: Path, date: str | None = None) -> Path:
    """Tar the sample tree to ``dest`` and make it readable by the login user."""
    root = Path(root)
    if date:
        source = root / date
        arcname = date
    else:
        source = root
        arcname = "wake-samples"
    wavs = list(source.rglob("*.wav")) if source.is_dir() else []
    if not wavs:
        raise SystemExit(f"No wake samples under {source}. Run `gva wake record` first.")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(source, arcname=arcname)
    os.chmod(temporary, 0o644)
    temporary.replace(dest)
    os.chmod(dest, 0o644)
    _give_to_login_user(dest)
    return dest


def _give_to_login_user(path: Path) -> None:
    """When this process is root via sudo, hand the archive to the person who asked."""
    user = os.environ.get("SUDO_USER")
    if not user or os.geteuid() != 0:
        return
    try:
        record = pwd.getpwnam(user)
    except KeyError:
        return
    os.chown(path, record.pw_uid, record.pw_gid)
