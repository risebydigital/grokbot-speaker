# Milo wake model

Binary openWakeWord classifier for **milo** and **hey milo**. The file is ONNX
because the Pi runs Python 3.13, which has no `tflite-runtime` wheel.

## Training

* Pipeline: [openWakeWord automatic training](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb) (`openwakeword/train.py`).
* Steps: 8000 on the first cosine cycle, then 800 and another 800. Checkpoint averaging is the stock procedure. Negative-example weight rises from 1 to 200 and is not doubled between sequences: the stock tracker never updates its false-positive count, so the doubling collapses a short CPU run into an all-negative model. Each batch has 48 positives, 64 adversarial negatives (confusables and phoneme overlap), and 128 generic ACAV windows. This is shorter than the notebook's 50,000-step default so it finishes on CPU.
* Classifier: fully connected, layer size 64, input `(batch, 16, 96)` from the v0.5.1 embedding model (the same ONNX file the runtime downloads).
* Positive clips: about 655 training and 234 validation, half "milo" and half "hey milo". The positive directory can be a bit larger when an earlier generation pass added clips; every wav in that directory is augmented.
* Voices: Piper ONNX British voices (alan, alba, cori, northern English male, jenny, aru, vctk) and other English voices (lessac, amy, ryan, joe), plus the LibriTTS multi-speaker Piper checkpoint from piper-sample-generator. Speed uses Piper length scales 0.85, 1.0, and 1.15. Pitch is shifted by -2 to +2 semitones. LibriTTS also varies speaker interpolation and duration noise.
* Confusable negatives, synthesised on purpose: Miles, my low, hello, mellow, Milan, Mylo, and the same phrases with "hey" in front. The trainer also adds phoneme-overlap adversaries.
* Augmentation: openWakeWord's augmenter (EQ, distortion, pitch shift, noise, gain) mixed with MIT environmental room impulse responses (`davidscripka/MIT_environmental_impulse_responses`, 270 files) and 250 clips from AudioSet balanced shard 09 (the parquet that replaced `bal_train09.tar`). A Free Music Archive slice was requested and skipped: that dataset's zip cannot be read as a stream.
* Generic negatives: the first 1,000,000 windows of `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` (about 356 hours of the standard 2,000-hour set). The whole file is 17 GB; the prefix keeps the run on this machine.
* False-accept set: `validation_set_features.npy` from `davidscripka/openwakeword_features`.

Recall is measured by running the shipped ONNX file through openWakeWord's streaming scorer on the held-out synthetic clips, with the device rule of 2 frames over the threshold. False accepts use that same rule on the validation features (one score every 80 ms, then a 2 second refractory). Raw window hits per hour are listed beside that, because a single loud frame is not yet a wake. The synthetic clips are not a recording of the AIRHUG in the room, so the recall number is an upper bound on what the microphone will do. Tune it with `gva wake test`.

## Validation

Validation audio duration: 10.697 hours (481345 embedding frames at 80 ms).

| threshold | recall | confusable accept | false accepts / hour | raw windows / hour |
| --- | --- | --- | --- | --- |
| 0.30 | 0.799 | 0.179 | 30.758 | 154.0 |
| 0.35 | 0.795 | 0.179 | 27.205 | 138.2 |
| 0.40 | 0.786 | 0.171 | 23.559 | 123.4 |
| 0.45 | 0.773 | 0.171 | 21.783 | 112.3 |
| 0.50 | 0.756 | 0.167 | 20.287 | 102.6 |
| 0.55 | 0.752 | 0.158 | 18.230 | 91.5 |
| 0.60 | 0.748 | 0.158 | 16.360 | 82.2 |
| 0.65 | 0.714 | 0.158 | 14.023 | 72.8 |
| 0.70 | 0.692 | 0.154 | 12.714 | 64.2 |
| 0.75 | 0.675 | 0.145 | 11.405 | 56.5 |
| 0.80 | 0.662 | 0.128 | 9.723 | 48.1 |
| 0.85 | 0.641 | 0.111 | 7.292 | 37.5 |
| 0.90 | 0.620 | 0.103 | 5.048 | 26.6 |

Recommended threshold: **0.30**

At that threshold, recall is 0.799, false accepts per hour are 30.758, and the confusable-clip accept rate is 0.179.

No row clears the bar for replacing hey_jarvis. The **0.30** row is the widest gap between recall and confusable accept, and it is still too noisy to ship as the default. If you set `wake.model: milo`, start at **0.60** (recall 0.748, false accepts per hour 16.360, confusable accept 0.158) and move it with `gva wake test`.

Shipped default: `hey_jarvis`. The default stays `hey_jarvis` unless recall is at least 0.8 and false accepts per hour are at most 0.5, with confusable clips accepted no more than 2 percent of the time.

## Licence

The classifier weights were trained for this project. openWakeWord is Apache-2.0. The embedding and melspectrogram networks it depends on are the v0.5.1 ONNX files published with openWakeWord. The stock `hey_jarvis` model, used when this file is missing, is CC BY-NC-SA 4.0.
