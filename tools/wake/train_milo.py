#!/usr/bin/env python3
"""Train the shipped openWakeWord Milo model.

This follows openWakeWord's automatic training pipeline (``train.py``):
Piper synthesises the positive clips, the same generator builds phoneme
adversaries, clips are augmented with room impulse responses and background
noise, and a small fully connected classifier is trained on the frozen
openWakeWord embedding features.

Two Piper front ends are used:

* the piper-sample-generator LibriTTS model (many English speakers, with
  length-scale / noise variation and speaker interpolation)
* British and other English Piper ONNX voices (alan, alba, cori, northern
  English, jenny, aru, vctk, lessac, amy, ryan, joe), with speed, noise,
  and a pitch shift

The primary phrase is "hey milo". "milo" is one clip in five, so a bare
name still has a chance of scoring, but the model is aimed at the two-word
phrase. The negative set includes the confusable phrases Miles, my low,
hello, mellow, Milan, and Mylo, plus "hey" in front of each.

Real AIRHUG clips, once copied off the Pi, are mixed in with::

    /tmp/oww-venv/bin/python tools/wake/train_milo.py --train --real-samples /path/to/2026-10-26

Held-out real clips are not used for training. Their recall is printed
beside the synthetic numbers.

Run it with the training virtualenv after the datasets are on disk::

    /tmp/oww-venv/bin/python tools/wake/train_milo.py --train

The ONNX file is written to ``assets/wakewords/milo.onnx``. Metrics are
printed as JSON. ``--steps`` can be lowered when the machine is slow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWW = Path(os.environ.get("MILO_OWW_ROOT", "/tmp/oww"))
DATA = Path(os.environ.get("MILO_DATA", "/tmp/oww-data"))

# Four clips of the two-word phrase, then one bare "milo".
PHRASE_MIX = ("hey milo", "hey milo", "hey milo", "hey milo", "milo")
PHRASE_MIX_ID = "hey-milo-primary-v1"
CONFUSABLES = (
    "miles",
    "my low",
    "hello",
    "mellow",
    "milan",
    "mylo",
    "hey miles",
    "hey my low",
    "hey hello",
    "hey mellow",
    "hey milan",
    "hey mylo",
)

# ONNX voices. speaker counts come from each voice's JSON.
BRITISH_VOICES = (
    "en_GB-alan-medium",
    "en_GB-alba-medium",
    "en_GB-cori-medium",
    "en_GB-northern_english_male-medium",
    "en_GB-jenny_dioco-medium",
    "en_GB-aru-medium",
    "en_GB-vctk-medium",
)
OTHER_ENGLISH_VOICES = (
    "en_US-lessac-medium",
    "en_US-amy-medium",
    "en_US-ryan-high",
    "en_US-joe-medium",
)
LENGTH_SCALES = (0.85, 1.0, 1.15)
NOISE_SCALES = (0.5, 0.667, 0.9)
PITCHES = (-2.0, -1.0, 0.0, 1.0, 2.0)
TOTAL_LENGTH = 32000  # 2 seconds at 16 kHz -> 16 embedding frames


def _piper_binary() -> Path:
    found = sorted(DATA.joinpath("piper").rglob("piper"))
    binaries = [path for path in found if path.is_file() and os.access(path, os.X_OK)]
    if not binaries:
        raise SystemExit(f"Piper binary not found under {DATA / 'piper'}")
    return binaries[0]


def _piper_flag_names(piper: Path) -> dict[str, str]:
    help_text = subprocess.check_output([str(piper), "--help"], text=True, stderr=subprocess.STDOUT)

    def pick(*options: str) -> str:
        for option in options:
            if option in help_text:
                return option
        raise SystemExit(f"piper --help has none of {options}")

    return {
        "model": pick("-m", "--model"),
        "output_dir": pick("--output_dir", "--output-dir", "-d"),
        "length": pick("--length_scale", "--length-scale"),
        "noise": pick("--noise_scale", "--noise-scale"),
        "noise_w": pick("--noise_w", "--noise-w"),
        "speaker": pick("--speaker", "-s"),
    }


def _voice_speakers(model: Path) -> list[int]:
    config = json.loads(model.with_suffix(model.suffix + ".json").read_text(encoding="utf-8"))
    count = int(config.get("num_speakers") or 1)
    if count <= 1:
        return [0]
    # A spread of speakers, not all 109 VCTK ids. Enough accent variety, fewer model loads.
    step = max(1, count // 12)
    return list(range(0, count, step))[:12]


def _pitch_shift(samples, semitones: float):
    import numpy as np
    from scipy import signal

    if abs(semitones) < 0.05 or samples.size < 16:
        return samples
    factor = 2.0 ** (semitones / 12.0)
    floated = samples.astype(np.float32)
    pitched_len = max(16, int(round(floated.size / factor)))
    pitched = signal.resample(floated, pitched_len)
    restored = signal.resample(pitched, floated.size)
    return np.clip(np.round(restored), -32768, 32767).astype(np.int16)


def _to_16k(path: Path, dest: Path, semitones: float) -> None:
    import numpy as np
    from scipy import signal
    from scipy.io import wavfile

    rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if rate != 16000:
        count = max(16, int(round(audio.size * 16000 / rate)))
        audio = signal.resample(audio, count)
    audio = np.clip(np.round(audio), -32768, 32767).astype(np.int16)
    audio = _pitch_shift(audio, semitones)
    # Keep the word inside the 2 second training window.
    if audio.size > 28000:
        audio = signal.resample(audio.astype(np.float32), 26000)
        audio = np.clip(np.round(audio), -32768, 32767).astype(np.int16)
    if audio.size < 1600 or int(np.max(np.abs(audio))) < 200:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(dest, 16000, audio)


def synthesise_piper(output_dir: Path, phrases: tuple[str, ...] | list[str], per_voice: int, stem: str) -> int:
    """Write 16 kHz clips with the ONNX Piper voices. Returns how many files were written."""
    import itertools

    piper = _piper_binary()
    flags = _piper_flag_names(piper)
    variation = list(itertools.product(LENGTH_SCALES, NOISE_SCALES, PITCHES))
    serial = 0
    for voice_name in BRITISH_VOICES + OTHER_ENGLISH_VOICES:
        model = DATA / "voices" / f"{voice_name}.onnx"
        if not model.is_file():
            print(f"skip missing voice {voice_name}", file=sys.stderr)
            continue
        speakers = _voice_speakers(model)
        clips_per_speaker = max(1, per_voice // len(speakers))
        for speaker in speakers:
            grouped: dict[tuple[float, float], list[tuple[str, float]]] = {}
            for index in range(clips_per_speaker):
                phrase = phrases[index % len(phrases)]
                length, noise, pitch = variation[(speaker + index) % len(variation)]
                grouped.setdefault((length, noise), []).append((phrase, pitch))
            for (length, noise), items in grouped.items():
                work = output_dir / f".piper-{stem}-{serial}"
                if work.exists():
                    shutil.rmtree(work)
                work.mkdir(parents=True)
                text = "\n".join(phrase for phrase, _pitch in items) + "\n"
                command = [
                    str(piper),
                    flags["model"],
                    str(model),
                    flags["output_dir"],
                    str(work),
                    flags["length"],
                    str(length),
                    flags["noise"],
                    str(noise),
                    flags["noise_w"],
                    "0.8",
                    flags["speaker"],
                    str(speaker),
                ]
                env = os.environ.copy()
                lib = str(piper.parent)
                env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
                subprocess.run(command, input=text, text=True, check=True, env=env, stdout=subprocess.DEVNULL)
                produced = sorted(work.glob("*.wav"))
                for wav_path, (_phrase, pitch) in zip(produced, items):
                    dest = output_dir / f"{stem}_{voice_name}_s{speaker}_{serial}.wav"
                    serial += 1
                    _to_16k(wav_path, dest, pitch)
                shutil.rmtree(work, ignore_errors=True)
    return len(list(output_dir.glob(f"{stem}_*.wav")))


def _download(url: str, dest: Path) -> None:
    import urllib.request

    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {dest.name}", flush=True)
    urllib.request.urlretrieve(url, dest)


def prepare_backgrounds() -> tuple[list[Path], list[Path]]:
    """MIT room impulses plus AudioSet and a short FMA slice. Returns (rir dirs, noise dirs)."""
    import numpy as np
    from scipy.io import wavfile

    rir_dir = DATA / "mit_rirs"
    rir_dir.mkdir(parents=True, exist_ok=True)
    if len(list(rir_dir.glob("*.wav"))) < 20:
        import datasets

        stream = datasets.load_dataset(
            "davidscripka/MIT_environmental_impulse_responses",
            split="train",
            streaming=True,
        )
        for index, row in enumerate(stream):
            audio = row["audio"]
            samples = np.asarray(audio["array"], dtype=np.float32)
            rate = int(audio["sampling_rate"])
            if rate != 16000 and samples.size:
                count = max(16, int(round(samples.size * 16000 / rate)))
                from scipy import signal

                samples = signal.resample(samples, count).astype(np.float32)
            name = Path(str(audio.get("path") or f"rir_{index}.wav")).name
            wavfile.write(rir_dir / name.replace(".flac", ".wav"), 16000, np.clip(samples, -1, 1))
            if index >= 399:
                break
        print(f"room impulses: {len(list(rir_dir.glob('*.wav')))}", flush=True)

    noise_dir = DATA / "audioset_16k"
    noise_dir.mkdir(parents=True, exist_ok=True)
    if len(list(noise_dir.glob("*.wav"))) < 30:
        import io

        import pyarrow.parquet as pq
        import soundfile as sf
        from huggingface_hub import hf_hub_download
        from scipy import signal

        # The notebook's bal_train09.tar is gone. The same shard is a parquet of flac bytes.
        parquet = hf_hub_download(
            "agkphysics/AudioSet",
            "data/bal_train/09.parquet",
            repo_type="dataset",
        )
        count = 0
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1):
            audio = batch.to_pydict()["audio"][0]
            samples, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
            samples = samples.mean(axis=1)
            if rate != 16000 and samples.size:
                samples = signal.resample(samples, max(16, int(round(samples.size * 16000 / rate)))).astype("float32")
            samples = samples[: 16000 * 5]
            if samples.size < 1600:
                continue
            wavfile.write(noise_dir / f"audioset_{count:04d}.wav", 16000, np.clip(samples, -1, 1))
            count += 1
            if count >= 250:
                break
        print(f"audioset clips: {count}", flush=True)

    music_dir = DATA / "fma"
    music_dir.mkdir(parents=True, exist_ok=True)
    if len(list(music_dir.glob("*.wav"))) < 8:
        try:
            import datasets
            from scipy import signal

            stream = datasets.load_dataset(
                "rudraml/fma",
                name="small",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            stream = stream.cast_column("audio", datasets.Audio(sampling_rate=16000))
            saved = 0
            for row in stream:
                audio = row["audio"]
                samples = np.asarray(audio["array"], dtype=np.float32)
                if samples.ndim > 1:
                    samples = samples.mean(axis=1)
                samples = samples[: 16000 * 15]
                if samples.size < 16000:
                    continue
                name = Path(str(audio.get("path") or f"fma_{saved}.mp3")).stem + ".wav"
                wavfile.write(music_dir / name, 16000, np.clip(samples, -1, 1))
                saved += 1
                if saved >= 12:
                    break
            print(f"fma clips: {saved}", flush=True)
        except Exception as exc:
            print(f"FMA download skipped: {exc.__class__.__name__}: {exc}", file=sys.stderr)
    rir_dirs = [rir_dir]
    noise_dirs = [path for path in (noise_dir, music_dir) if any(path.glob("*.wav"))]
    if not noise_dirs:
        raise SystemExit("No background audio was downloaded")
    if not any(rir_dir.glob("*.wav")):
        raise SystemExit("No room impulse responses were downloaded")
    return rir_dirs, noise_dirs


def write_config(
    config_path: Path,
    output_dir: Path,
    n_samples: int,
    n_val: int,
    steps: int,
    rir_dirs: list[Path],
    noise_dirs: list[Path],
) -> None:
    import yaml

    custom: list[str] = []
    for phrase in CONFUSABLES:
        custom.extend([phrase] * 6)
    config = {
        "model_name": "milo",
        "target_phrase": list(PHRASE_MIX),
        "custom_negative_phrases": custom,
        "n_samples": n_samples,
        "n_samples_val": n_val,
        "tts_batch_size": 16,
        "augmentation_batch_size": 8,
        "piper_sample_generator_path": str(OWW / "piper-sample-generator"),
        "output_dir": str(output_dir),
        "rir_paths": [str(path) for path in rir_dirs],
        "background_paths": [str(path) for path in noise_dirs],
        "background_paths_duplication_rate": [1] * len(noise_dirs),
        "false_positive_validation_data_path": str(DATA / "validation_set_features.npy"),
        "augmentation_rounds": 1,
        "feature_data_files": {
            "ACAV100M_sample": str(DATA / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"),
        },
        "batch_n_per_class": {
            "ACAV100M_sample": 128,
            "adversarial_negative": 64,
            "positive": 48,
        },
        "model_type": "dnn",
        "layer_size": 64,
        "steps": steps,
        "max_negative_weight": 200,
        "target_false_positives_per_hour": 0.2,
        "total_length": TOTAL_LENGTH,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def patch_piper_generator() -> None:
    """PyTorch 2.6 refuses the trusted Piper checkpoint unless weights_only is off."""
    path = OWW / "piper-sample-generator" / "generate_samples.py"
    text = path.read_text(encoding="utf-8")
    old = "    model = torch.load(model_path)\n"
    new = "    model = torch.load(model_path, weights_only=False)\n"
    if old not in text:
        if "weights_only=False" in text:
            return
        raise SystemExit("piper generate_samples.py no longer matches the torch.load line")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _trainer_followups(text: str) -> str:
    """Patches that stay correct if train.py was already patched once."""
    text = text.replace(
        "        if args.convert_to_tflite:\n",
        "        if args.convert_to_tflite is True:\n",
        1,
    )
    doubled = (
        "        if self.best_val_fp > target_fp_per_hour:\n"
        "            max_negative_weight = max_negative_weight*2\n"
        '            logging.info("Increasing weight on negative examples to reduce false positives...")\n'
    )
    held = (
        "        # best_val_fp starts at 1000 and is never updated, so this always doubles\n"
        "        # and a short CPU run collapses to an all-negative classifier.\n"
        "        if False and self.best_val_fp > target_fp_per_hour:\n"
        "            max_negative_weight = max_negative_weight*2\n"
        '            logging.info("Increasing weight on negative examples to reduce false positives...")\n'
    )
    text = text.replace(doubled, held)
    empty = "        else:\n            combined_model = self.model\n"
    kept = (
        "        else:\n"
        "            if self.best_models:\n"
        "                _best = int(np.argmax([score['val_recall'] for score in self.best_model_scores]))\n"
        "                combined_model = self.best_models[_best]\n"
        "                logging.info('Percentile merge was empty. Using the checkpoint with the highest validation recall.')\n"
        "            else:\n"
        "                combined_model = self.model\n"
    )
    if empty in text and "Percentile merge was empty" not in text:
        text = text.replace(empty, kept, 1)
    static_export = '        _export_kwargs = {"opset_version": 13}\n'
    dynamic_export = (
        "        _export_kwargs = {\n"
        '            "opset_version": 13,\n'
        '            "input_names": ["input"],\n'
        '            "output_names": ["output"],\n'
        '            "dynamic_axes": {"input": {0: "batch"}, "output": {0: "batch"}},\n'
        "        }\n"
    )
    if static_export in text and "dynamic_axes" not in text:
        text = text.replace(static_export, dynamic_export, 1)
    return text


def patch_trainer(train_py: Path) -> None:
    """Keep the official loop, but make it finish on CPU and not allocate the whole FP set."""
    text = train_py.read_text(encoding="utf-8")
    if "GVA_MILO_PATCH" in text:
        updated = _trainer_followups(text)
        if updated != text:
            train_py.write_text(updated, encoding="utf-8")
        return
    text = text.replace(
        "if __name__ == '__main__':\n",
        "if __name__ == '__main__':\n"
        "    # GVA_MILO_PATCH\n"
        "    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')\n",
        1,
    )
    text = text.replace("        steps = steps/10\n", "        steps = max(1, int(steps) // 10)\n", 1)
    forced_length = (
        "    config[\"total_length\"] = int(round(np.median(duration_in_samples)/1000)*1000) + 12000  # add 750 ms to clip duration as buffer\n"
        "    if config[\"total_length\"] < 32000:\n"
        "        config[\"total_length\"] = 32000  # set a minimum of 32000 samples (2 seconds)\n"
        "    elif abs(config[\"total_length\"] - 32000) <= 4000:\n"
        "        config[\"total_length\"] = 32000\n"
    )
    if forced_length not in text:
        raise SystemExit("openWakeWord train.py no longer matches the clip-length block")
    text = text.replace(
        forced_length,
        "    # GVA_MILO_PATCH: the runtime scores fixed 16-frame (2 second) windows.\n"
        "    config[\"total_length\"] = 32000\n",
        1,
    )
    old_transform = (
        "            else:\n"
        "                return x\n"
        "            return new_batch\n"
    )
    new_transform = (
        "            else:\n"
        "                new_batch = x\n"
        "            # ACAV features are float16. CPU Linear does not accept Half.\n"
        "            return np.ascontiguousarray(new_batch, dtype=np.float32)\n"
    )
    if old_transform not in text:
        raise SystemExit("openWakeWord train.py no longer matches the feature transform")
    text = text.replace(old_transform, new_transform, 1)
    text = text.replace(
        "        if step_ndx == max_steps-1:\n",
        "        if step_ndx >= int(max_steps) - 1:\n",
        1,
    )
    text = text.replace(
        "        # Move models and main class to target device\n",
        "        max_steps = int(max_steps)\n"
        "        warmup_steps = int(warmup_steps)\n"
        "        hold_steps = int(hold_steps)\n"
        "        # Move models and main class to target device\n",
        1,
    )
    text = text.replace(
        "        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),\n"
        "                                              batch_size=None, num_workers=n_cpus, prefetch_factor=16)\n",
        "        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),\n"
        "                                              batch_size=None, num_workers=0)\n",
        1,
    )
    old_fp = """        X_val_fp = np.load(config["false_positive_validation_data_path"])
        X_val_fp = np.array([X_val_fp[i:i+input_shape[0]] for i in range(0, X_val_fp.shape[0]-input_shape[0], 1)])  # reshape to match model
        X_val_fp_labels = np.zeros(X_val_fp.shape[0]).astype(np.float32)
        X_val_fp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),
            batch_size=len(X_val_fp_labels)
        )
"""
    new_fp = '''        class _FpWindows(torch.utils.data.Dataset):
            """Window the FP set on the fly so the 11 hour file is not copied into RAM."""

            def __init__(self, features, window, stride=8):
                # Stride keeps checkpoint selection on the full file without scoring every 80 ms frame.
                # The reported false-accept rate is computed afterwards at stride 1.
                self.features = features
                self.window = int(window)
                self.stride = int(stride)
                if features.ndim == 3:
                    self.n = int(features.shape[0])
                    self.ready = True
                else:
                    self.n = (int(features.shape[0]) - self.window) // self.stride + 1
                    self.ready = False

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                if self.ready:
                    window = self.features[idx]
                else:
                    start = idx * self.stride
                    window = self.features[start:start + self.window]
                tensor = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))
                return tensor, torch.tensor(0.0, dtype=torch.float32)

        _fp_source = np.load(config["false_positive_validation_data_path"], mmap_mode="r")
        X_val_fp = torch.utils.data.DataLoader(
            _FpWindows(_fp_source, input_shape[0]),
            batch_size=2048,
            shuffle=False,
            num_workers=0,
        )
'''
    if old_fp not in text:
        raise SystemExit("openWakeWord train.py no longer matches the FP-loader patch")
    text = text.replace(old_fp, new_fp, 1)
    old_export = (
        "        torch.onnx.export(model_to_save.to(\"cpu\"), torch.rand(self.input_shape)[None, ],\n"
        "                          os.path.join(output_dir, model_name + \".onnx\"), opset_version=13)\n"
    )
    new_export = (
        "        _export_kwargs = {\"opset_version\": 13}\n"
        "        try:\n"
        "            import inspect as _inspect\n"
        "            if \"dynamo\" in _inspect.signature(torch.onnx.export).parameters:\n"
        "                _export_kwargs[\"dynamo\"] = False\n"
        "        except Exception:\n"
        "            pass\n"
        "        torch.onnx.export(model_to_save.to(\"cpu\"), torch.rand(self.input_shape)[None, ],\n"
        "                          os.path.join(output_dir, model_name + \".onnx\"), **_export_kwargs)\n"
    )
    if old_export not in text:
        raise SystemExit("openWakeWord train.py no longer matches the ONNX export patch")
    text = text.replace(old_export, new_export, 1)
    text = _trainer_followups(text)
    train_py.write_text(text, encoding="utf-8")


def _count_wavs(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return len(list(directory.glob("*.wav")))


def probe_libritts(seconds_budget: float) -> int:
    """How many extra LibriTTS clips fit in the budget. 0 if the model is too slow."""
    sys.path.insert(0, str(OWW / "piper-sample-generator"))
    from generate_samples import generate_samples

    probe = DATA / "libritts_probe"
    if probe.exists():
        shutil.rmtree(probe)
    probe.mkdir()
    started = time.monotonic()
    generate_samples(
        text=list(PHRASE_MIX),
        max_samples=4,
        batch_size=2,
        noise_scales=[0.8],
        noise_scale_ws=[0.8],
        length_scales=[0.85, 1.0, 1.15],
        output_dir=str(probe),
        auto_reduce_batch_size=False,
        max_speakers=80,
    )
    elapsed = time.monotonic() - started
    per_clip = elapsed / 4
    print(f"libritts probe {elapsed:.1f}s for 4 clips ({per_clip:.2f}s each)", flush=True)
    affordable = int(seconds_budget / max(per_clip, 0.05))
    shutil.rmtree(probe, ignore_errors=True)
    return max(0, affordable)


def run_trainer(config_path: Path, *flags: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(OWW / "openwakeword") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("OMP_NUM_THREADS", "2")
    command = [sys.executable, str(OWW / "openwakeword" / "openwakeword" / "train.py"), "--training_config", str(config_path), *flags]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=env, cwd=str(OWW))


def _session(model_path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def _score_matrix(session, windows) -> list[float]:
    import numpy as np

    name = session.get_inputs()[0].name
    result = session.run(None, {name: np.ascontiguousarray(windows, dtype=np.float32)})[0]
    return np.asarray(result).reshape(-1).tolist()


def _clip_fires(scores: list[float], threshold: float, patience: int = 2) -> bool:
    above = 0
    for score in scores:
        if score >= threshold:
            above += 1
            if above >= patience:
                return True
        else:
            above = 0
    return False


def _false_accepts(scores: list[float], threshold: float, patience: int = 2, refractory: int = 25) -> int:
    above = 0
    cool = 0
    hits = 0
    for score in scores:
        if cool:
            cool -= 1
            above = 0
            continue
        if score >= threshold:
            above += 1
            if above >= patience:
                hits += 1
                above = 0
                cool = refractory
        else:
            above = 0
    return hits


def evaluate(model_path: Path, feature_dir: Path, positive_wavs: list[Path], negative_wavs: list[Path]) -> dict:
    import numpy as np

    session = _session(model_path)
    positive = np.load(feature_dir / "positive_features_test.npy", mmap_mode="r")
    negative = np.load(feature_dir / "negative_features_test.npy", mmap_mode="r")
    pos_scores = _score_matrix(session, positive)
    neg_scores = _score_matrix(session, negative)

    fp = np.load(DATA / "validation_set_features.npy", mmap_mode="r")
    window = 16
    hop_s = 0.08
    hours = float(fp.shape[0]) * hop_s / 3600.0
    # Score the streaming windows in chunks. Stride 1 matches the runtime (one score per 80 ms).
    fp_scores: list[float] = []
    count = int(fp.shape[0]) - window + 1
    for start in range(0, count, 4096):
        end = min(count, start + 4096)
        batch = np.stack([fp[index : index + window] for index in range(start, end)]).astype(np.float32)
        fp_scores.extend(_score_matrix(session, batch))

    # Runtime path: openWakeWord's own streaming scorer on the held-out wavs.
    sys.path.insert(0, str(OWW / "openwakeword"))
    from openwakeword.model import Model

    runtime = Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
    )

    def runtime_scores(paths: list[Path]) -> list[list[float]]:
        rows = []
        key = model_path.stem
        for path in paths:
            frames = runtime.predict_clip(str(path))
            rows.append([float(frame.get(key, 0.0)) for frame in frames])
            runtime.reset()
        return rows

    runtime_pos = runtime_scores(positive_wavs)
    runtime_neg = runtime_scores(negative_wavs)

    thresholds = [round(value, 2) for value in np.arange(0.30, 0.91, 0.05)]
    rows = []
    for threshold in thresholds:
        feature_recall = float(np.mean([score >= threshold for score in pos_scores])) if pos_scores else 0.0
        feature_confusable = float(np.mean([score >= threshold for score in neg_scores])) if neg_scores else 0.0
        raw_fp = float(sum(score >= threshold for score in fp_scores) / hours) if hours else 0.0
        device_fp = _false_accepts(fp_scores, threshold) / hours if hours else 0.0
        recall = (
            float(np.mean([_clip_fires(scores, threshold) for scores in runtime_pos])) if runtime_pos else 0.0
        )
        confusable = (
            float(np.mean([_clip_fires(scores, threshold) for scores in runtime_neg])) if runtime_neg else 0.0
        )
        rows.append(
            {
                "threshold": threshold,
                "recall": round(recall, 4),
                "confusable_accept_rate": round(confusable, 4),
                "feature_recall": round(feature_recall, 4),
                "feature_confusable_accept_rate": round(feature_confusable, 4),
                "false_accepts_per_hour": round(device_fp, 4),
                "raw_windows_per_hour": round(raw_fp, 4),
            }
        )
    eligible = [
        row
        for row in rows
        if row["recall"] >= 0.8 and row["confusable_accept_rate"] <= 0.02
    ]
    if eligible:
        recommended = max(eligible, key=lambda row: row["threshold"])
    else:
        # No threshold clears both gates. Prefer the point that separates
        # the wake phrase from the confusable clips, then the quieter room.
        recommended = max(
            rows,
            key=lambda row: (
                row["recall"] - row["confusable_accept_rate"],
                -row["false_accepts_per_hour"],
                row["threshold"],
            ),
        )
    low_fp = recommended["false_accepts_per_hour"] <= 0.5
    use_as_default = bool(eligible) and low_fp and recommended["recall"] >= 0.8
    return {
        "positive_clips": len(positive_wavs),
        "negative_clips": len(negative_wavs),
        "feature_positive_rows": int(positive.shape[0]),
        "feature_negative_rows": int(negative.shape[0]),
        "validation_hours": round(hours, 3),
        "validation_frames": int(fp.shape[0]),
        "sweep": rows,
        "recommended_threshold": recommended["threshold"],
        "recommended_recall": recommended["recall"],
        "recommended_false_accepts_per_hour": recommended["false_accepts_per_hour"],
        "recommended_confusable_accept_rate": recommended["confusable_accept_rate"],
        "use_as_default": use_as_default,
    }


def reset_for_phrase_mix(model_dir: Path) -> None:
    """Drop clips and features when they were built for a different phrase mix.

    The trainer skips synthesis once a directory already holds enough wavs, so an
    older half-and-half "milo" / "hey milo" set would be reused forever.
    """
    marker = model_dir / "phrase_mix.txt"
    current = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    if current == PHRASE_MIX_ID:
        return
    print(f"phrase mix {current or 'unset'} -> {PHRASE_MIX_ID}; clearing clips and features", flush=True)
    for name in ("positive_train", "positive_test", "negative_train", "negative_test"):
        directory = model_dir / name
        if not directory.is_dir():
            continue
        for path in directory.glob("*.wav"):
            path.unlink()
    for path in model_dir.glob("*.npy"):
        path.unlink()
    marker.write_text(PHRASE_MIX_ID + "\n", encoding="utf-8")


def _holdout_split(paths: list[Path], fraction: float = 0.2) -> tuple[list[Path], list[Path]]:
    """Last slice is held out. Fewer than five clips all stay in training."""
    ordered = sorted(paths)
    if len(ordered) < 5:
        return ordered, []
    hold = max(1, int(round(len(ordered) * fraction)))
    hold = min(hold, len(ordered) - 1)
    return ordered[:-hold], ordered[-hold:]


def resolve_real_root(path: Path) -> Path:
    """A date folder, or the latest date folder under a sample root."""
    path = Path(path)
    if not path.is_dir():
        raise SystemExit(f"No real samples at {path}")
    if any((path / name).is_dir() for name in ("positive", "negative", "background")):
        return path
    dated = sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and any((child / name).is_dir() for name in ("positive", "negative"))
    )
    if not dated:
        raise SystemExit(f"No positive/ or negative/ under {path}")
    chosen = dated[-1]
    print(f"using real samples in {chosen.name}", flush=True)
    return chosen


def mix_real_samples(model_dir: Path, real_root: Path) -> dict:
    """Copy the training split in beside the synthetic clips and keep a holdout."""
    positive = sorted((real_root / "positive").glob("*.wav")) if (real_root / "positive").is_dir() else []
    negative = sorted((real_root / "negative").glob("*.wav")) if (real_root / "negative").is_dir() else []
    if not positive and not negative:
        raise SystemExit(f"No wav files in {real_root}/positive or {real_root}/negative")
    pos_train, pos_hold = _holdout_split(positive)
    neg_train, neg_hold = _holdout_split(negative)
    for source in pos_train:
        shutil.copyfile(source, model_dir / "positive_train" / f"real_{source.name}")
    for source in neg_train:
        shutil.copyfile(source, model_dir / "negative_train" / f"real_{source.name}")
    for path in model_dir.glob("*.npy"):
        path.unlink()
    background = real_root / "background"
    has_background = background.is_dir() and any(background.glob("*.wav"))
    print(
        f"real samples train {len(pos_train)} positive / {len(neg_train)} negative; "
        f"held out {len(pos_hold)} positive / {len(neg_hold)} negative",
        flush=True,
    )
    return {
        "used": True,
        "train_positive": len(pos_train),
        "train_negative": len(neg_train),
        "holdout_positive": len(pos_hold),
        "holdout_negative": len(neg_hold),
        "background": has_background,
        "background_dir": background if has_background else None,
        "holdout_positive_paths": pos_hold,
        "holdout_negative_paths": neg_hold,
    }


def score_real_clips(model_path: Path, positive_wavs: list[Path], negative_wavs: list[Path]) -> list[dict]:
    """Patience-2 streaming scores on held-out real wavs. Same rule as the device."""
    if not positive_wavs and not negative_wavs:
        return []
    import numpy as np

    sys.path.insert(0, str(OWW / "openwakeword"))
    from openwakeword.model import Model

    runtime = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
    key = model_path.stem

    def rows_for(paths: list[Path]) -> list[list[float]]:
        rows = []
        for path in paths:
            frames = runtime.predict_clip(str(path))
            rows.append([float(frame.get(key, 0.0)) for frame in frames])
            runtime.reset()
        return rows

    positive_scores = rows_for(positive_wavs)
    negative_scores = rows_for(negative_wavs)
    thresholds = [round(float(value), 2) for value in np.arange(0.30, 0.91, 0.05)]
    rows = []
    for threshold in thresholds:
        recall = (
            float(np.mean([_clip_fires(scores, threshold) for scores in positive_scores]))
            if positive_scores
            else None
        )
        confusable = (
            float(np.mean([_clip_fires(scores, threshold) for scores in negative_scores]))
            if negative_scores
            else None
        )
        rows.append(
            {
                "threshold": threshold,
                "recall": None if recall is None else round(recall, 4),
                "confusable_accept_rate": None if confusable is None else round(confusable, 4),
            }
        )
    return rows


def format_real_block(info: dict | None) -> str:
    if not info or not info.get("used"):
        return (
            "## Real clips\n\n"
            "This run did not include AIRHUG recordings. On the Pi, `sudo gva wake record` "
            "then `sudo gva wake export`. Copy the archive off the Pi, unpack it, and train "
            "again with `--real-samples` pointing at the date folder. Real positives are "
            "augmented with the synthetic clips. The last fifth of each real class (when "
            "there are at least five clips) is held out and scored with the same two-frame rule.\n"
        )
    lines = [
        "## Real clips",
        "",
        (
            f"Real AIRHUG takes mixed into training: {info['train_positive']} positive and "
            f"{info['train_negative']} negative. They are augmented with the same room impulse "
            "responses and noise as the synthetic clips. The ACAV negatives and the synthetic "
            "confusable set stay in the batch."
        ),
        "",
    ]
    if info["holdout_positive"] == 0 and info["holdout_negative"] == 0:
        lines.append(
            "Fewer than five real clips in each class, so every real clip stayed in training. "
            "There is no held-out real score for this run."
        )
    else:
        lines.append(
            f"Held out of training: {info['holdout_positive']} positive and "
            f"{info['holdout_negative']} negative. Scored with two frames over the threshold."
        )
        lines.append("")
        lines.append("| threshold | real recall | real confusable accept |")
        lines.append("| --- | --- | --- |")
        for row in info.get("sweep") or []:
            recall = "n/a" if row["recall"] is None else f"{row['recall']:.3f}"
            confusable = "n/a" if row["confusable_accept_rate"] is None else f"{row['confusable_accept_rate']:.3f}"
            lines.append(f"| {row['threshold']:.2f} | {recall} | {confusable} |")
        chosen = info.get("at_recommended")
        if chosen:
            recall = "n/a" if chosen["recall"] is None else f"{chosen['recall']:.3f}"
            confusable = "n/a" if chosen["confusable_accept_rate"] is None else f"{chosen['confusable_accept_rate']:.3f}"
            lines.append("")
            lines.append(
                f"At the recommended threshold {chosen['threshold']:.2f}, held-out real recall is {recall} "
                f"and held-out real confusable accept is {confusable}."
            )
    lines.append("")
    return "\n".join(lines)


def write_model_card(
    metrics: dict, steps: int, n_samples: int, n_val: int, destination: Path, real_block: str
) -> None:
    sweep = "\n".join(
        f"| {row['threshold']:.2f} | {row['recall']:.3f} | {row['confusable_accept_rate']:.3f} | "
        f"{row['false_accepts_per_hour']:.3f} | {row['raw_windows_per_hour']:.1f} |"
        for row in metrics["sweep"]
    )
    default = "milo" if metrics["use_as_default"] else "hey_jarvis"
    trial = min(metrics["sweep"], key=lambda row: abs(row["threshold"] - 0.6))
    if metrics["use_as_default"]:
        advice = ""
    else:
        advice = (
            "No row clears the bar for replacing hey_jarvis. "
            f"The **{metrics['recommended_threshold']:.2f}** row is the widest gap between recall and confusable accept, "
            "and it is still too noisy to ship as the default. "
            f"If you set `wake.model: milo`, start at **{trial['threshold']:.2f}** "
            f"(recall {trial['recall']:.3f}, false accepts per hour {trial['false_accepts_per_hour']:.3f}, "
            f"confusable accept {trial['confusable_accept_rate']:.3f}) and move it with `gva wake test`.\n"
        )
    text = f"""# Milo wake model

Binary openWakeWord classifier for **hey milo**. **milo** is a secondary phrase,
about one synthetic clip in five. The file is ONNX because the Pi runs
Python 3.13, which has no `tflite-runtime` wheel.

## Training

* Pipeline: [openWakeWord automatic training](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb) (`openwakeword/train.py`).
* Steps: {steps} on the first cosine cycle, then {max(1, steps // 10)} and another {max(1, steps // 10)}. Checkpoint averaging is the stock procedure. Negative-example weight rises from 1 to 200 and is not doubled between sequences: the stock tracker never updates its false-positive count, so the doubling collapses a short CPU run into an all-negative model. Each batch has 48 positives, 64 adversarial negatives (confusables and phoneme overlap), and 128 generic ACAV windows. This is shorter than the notebook's 50,000-step default so it finishes on CPU.
* Classifier: fully connected, layer size 64, input `(batch, 16, 96)` from the v0.5.1 embedding model (the same ONNX file the runtime downloads).
* Positive clips: about {n_samples} training and {n_val} validation. Four out of five are "hey milo". "milo" is the secondary phrase. Real AIRHUG takes, when `--real-samples` is set, are copied into the same directories and augmented with the synthetic clips. One in five of those real takes (when there are at least five) is held out and is not trained on.
* Voices: Piper ONNX British voices (alan, alba, cori, northern English male, jenny, aru, vctk) and other English voices (lessac, amy, ryan, joe), plus the LibriTTS multi-speaker Piper checkpoint from piper-sample-generator. Speed uses Piper length scales 0.85, 1.0, and 1.15. Pitch is shifted by -2 to +2 semitones. LibriTTS also varies speaker interpolation and duration noise.
* Confusable negatives, synthesised on purpose: Miles, my low, hello, mellow, Milan, Mylo, and the same phrases with "hey" in front. The trainer also adds phoneme-overlap adversaries.
* Augmentation: openWakeWord's augmenter (EQ, distortion, pitch shift, noise, gain) mixed with MIT environmental room impulse responses (`davidscripka/MIT_environmental_impulse_responses`, 270 files) and 250 clips from AudioSet balanced shard 09 (the parquet that replaced `bal_train09.tar`). A Free Music Archive slice was requested and skipped: that dataset's zip cannot be read as a stream.
* Generic negatives: the first 1,000,000 windows of `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` (about 356 hours of the standard 2,000-hour set). The whole file is 17 GB; the prefix keeps the run on this machine.
* False-accept set: `validation_set_features.npy` from `davidscripka/openwakeword_features`.

Recall is measured by running the shipped ONNX file through openWakeWord's streaming scorer on the held-out synthetic clips, with the device rule of 2 frames over the threshold. False accepts use that same rule on the validation features (one score every 80 ms, then a 2 second refractory). Raw window hits per hour are listed beside that, because a single loud frame is not yet a wake. The synthetic clips are not a recording of the AIRHUG in the room, so the recall number is an upper bound on what the microphone will do. Tune it with `gva wake test`.

## Validation

Validation audio duration: {metrics['validation_hours']} hours ({metrics['validation_frames']} embedding frames at 80 ms).

| threshold | recall | confusable accept | false accepts / hour | raw windows / hour |
| --- | --- | --- | --- | --- |
{sweep}

Recommended threshold: **{metrics['recommended_threshold']:.2f}**

At that threshold, recall is {metrics['recommended_recall']:.3f}, false accepts per hour are {metrics['recommended_false_accepts_per_hour']:.3f}, and the confusable-clip accept rate is {metrics['recommended_confusable_accept_rate']:.3f}.

{advice}
Shipped default: `{default}`. The default stays `hey_jarvis` unless recall is at least 0.8 and false accepts per hour are at most 0.5, with confusable clips accepted no more than 2 percent of the time.

{real_block}
## Licence

The classifier weights were trained for this project. openWakeWord is Apache-2.0. The embedding and melspectrogram networks it depends on are the v0.5.1 ONNX files published with openWakeWord. The stock `hey_jarvis` model, used when this file is missing, is CC BY-NC-SA 4.0.
"""
    destination.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Milo openWakeWord ONNX model.")
    parser.add_argument("--train", action="store_true", help="Run synthesis, augmentation, and training.")
    parser.add_argument("--steps", type=int, default=int(os.environ.get("MILO_STEPS", "4000")))
    parser.add_argument("--piper-per-voice", type=int, default=24, help="Training clips per ONNX voice, split across its speakers.")
    parser.add_argument("--piper-val-per-voice", type=int, default=6)
    parser.add_argument("--libritts-seconds", type=float, default=8 * 60, help="CPU time budget for the LibriTTS generator.")
    parser.add_argument(
        "--real-samples",
        type=Path,
        default=None,
        help="Date folder from `gva wake record`, or a root that contains date folders.",
    )
    args = parser.parse_args()
    if not args.train:
        print("Pass --train to build assets/wakewords/milo.onnx. See the module docstring.")
        return 0

    import torch

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(4)
    patch_piper_generator()
    output_dir = DATA / "milo_run"
    model_dir = output_dir / "milo"
    for name in ("positive_train", "positive_test", "negative_train", "negative_test"):
        (model_dir / name).mkdir(parents=True, exist_ok=True)
    reset_for_phrase_mix(model_dir)

    print("synthesising Piper ONNX clips", flush=True)
    if _count_wavs(model_dir / "positive_train") < 100:
        synthesise_piper(model_dir / "positive_train", PHRASE_MIX, args.piper_per_voice, "pos")
        synthesise_piper(model_dir / "positive_test", PHRASE_MIX, args.piper_val_per_voice, "pos")
        synthesise_piper(model_dir / "negative_train", CONFUSABLES, max(8, args.piper_per_voice // 2), "neg")
        synthesise_piper(model_dir / "negative_test", CONFUSABLES, max(4, args.piper_val_per_voice // 2), "neg")
    print(
        "piper clips",
        {name: _count_wavs(model_dir / name) for name in ("positive_train", "positive_test", "negative_train", "negative_test")},
        flush=True,
    )

    pos = _count_wavs(model_dir / "positive_train")
    neg = _count_wavs(model_dir / "negative_train")
    pos_val = _count_wavs(model_dir / "positive_test")
    neg_val = _count_wavs(model_dir / "negative_test")
    # A second run must not treat clips already on disk as a reason to synthesise another batch.
    if min(pos, neg) >= 500 and min(pos_val, neg_val) >= 150:
        n_samples = min(pos, neg)
        n_val = min(pos_val, neg_val)
        print(f"reusing clips n_samples={n_samples} n_val={n_val} steps={args.steps}", flush=True)
    else:
        extra = probe_libritts(args.libritts_seconds)
        # train.py generates n_samples positives and n_samples negatives, so the probe budget
        # is spent four times (train/val × positive/negative). Keep the total near the budget.
        extra_train = max(40, extra // 4)
        extra_val = max(20, extra // 10)
        n_samples = pos + extra_train
        n_val = pos_val + extra_val
        print(f"n_samples={n_samples} n_val={n_val} steps={args.steps}", flush=True)

    real_info = None
    if args.real_samples:
        real_root = resolve_real_root(args.real_samples)
        real_info = mix_real_samples(model_dir, real_root)
    rir_dirs, noise_dirs = prepare_backgrounds()
    if real_info and real_info.get("background_dir"):
        noise_dirs.append(real_info["background_dir"])
    config_path = output_dir / "milo.yaml"
    write_config(config_path, output_dir, n_samples, n_val, args.steps, rir_dirs, noise_dirs)
    patch_trainer(OWW / "openwakeword" / "openwakeword" / "train.py")

    # Confirm the 2 second window is exactly 16 embedding frames, which is what the runtime feeds.
    sys.path.insert(0, str(OWW / "openwakeword"))
    from openwakeword.utils import AudioFeatures

    shape = AudioFeatures(device="cpu").get_embedding_shape(TOTAL_LENGTH / 16000)
    if tuple(shape) != (16, 96):
        raise SystemExit(f"embedding shape {shape} is not (16, 96); refusing to train a model the runtime cannot score")

    run_trainer(config_path, "--generate_clips", "--augment_clips", "--train_model")
    produced = output_dir / "milo.onnx"
    if not produced.is_file():
        raise SystemExit(f"training did not write {produced}")

    positive_wavs = sorted((model_dir / "positive_test").glob("*.wav"))
    negative_wavs = sorted((model_dir / "negative_test").glob("*.wav"))
    metrics = evaluate(produced, model_dir, positive_wavs, negative_wavs)
    metrics["steps"] = args.steps
    metrics["n_samples"] = n_samples
    metrics["n_val"] = n_val
    metrics["phrase_mix"] = PHRASE_MIX_ID
    if real_info and real_info.get("used"):
        sweep = score_real_clips(
            produced,
            real_info["holdout_positive_paths"],
            real_info["holdout_negative_paths"],
        )
        real_info["sweep"] = sweep
        real_info["at_recommended"] = next(
            (row for row in sweep if row["threshold"] == metrics["recommended_threshold"]),
            None,
        )
        metrics["real"] = {
            "used": True,
            "train_positive": real_info["train_positive"],
            "train_negative": real_info["train_negative"],
            "holdout_positive": real_info["holdout_positive"],
            "holdout_negative": real_info["holdout_negative"],
            "background": real_info["background"],
            "sweep": sweep,
            "at_recommended": real_info["at_recommended"],
        }
    else:
        metrics["real"] = {"used": False}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in metrics if key != "sweep"}, indent=2), flush=True)

    asset_dir = ROOT / "assets" / "wakewords"
    asset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, asset_dir / "milo.onnx")
    write_model_card(
        metrics,
        args.steps,
        n_samples,
        n_val,
        asset_dir / "MODEL_CARD.md",
        format_real_block(real_info),
    )
    print(f"wrote {asset_dir / 'milo.onnx'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
