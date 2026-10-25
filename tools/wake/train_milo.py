#!/usr/bin/env python3
"""Plan a custom openWakeWord model for Milo.

openWakeWord has no pretrained Milo model. Training uses synthetic speech and
is too heavy for a Raspberry Pi 4. Run it on a laptop or in Colab, then copy
the ONNX file to the Pi.

Out of the box the speaker listens for "hey jarvis" (model hey_jarvis).
"""

from __future__ import annotations

import argparse
from pathlib import Path

NOTEBOOK = "https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb"
AUTOMATIC = "https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb"
SYNTHETIC = "https://github.com/dscripka/openWakeWord"

PLAN = """
Train a Milo wake word
======================

The Pi cannot train this model. Use a laptop with a GPU, or Colab:

  {notebook}

The automatic notebook (same method, checked into openWakeWord) is:

  {automatic}

Synthetic clips come from the openWakeWord training pipeline ({synthetic}),
which generates speech with a TTS model (Piper or VITS) and mixes noise.

What to train
-------------

The owner's wake word is Milo. Two-syllable names are easier to miss and
easier to false-trigger. Train both of these, and prefer the one that
behaves better in the room:

  1. hey milo     (recommended; three syllables)
  2. milo         (the name on its own)

Suggested notebook settings
---------------------------

  target phrase:        {phrase}
  model name:           {model_name}
  n_samples:            20000 positive, similar number of adversarial negatives
  false-activation set: include "hey jarvis", "milo's", "hello", "yellow"
  output:               {model_name}.onnx

Install the model on the Pi
----------------------------

  sudo install -m 644 {model_name}.onnx /usr/share/gva/wake/{model_name}.onnx
  sudo gva config set wake.model /usr/share/gva/wake/{model_name}.onnx
  sudo gva config set wake.phrase "{phrase}"
  sudo systemctl restart gva-voice

Until that file exists, config/config.example.yaml keeps:

  wake.model: hey_jarvis
  wake.phrase: "hey jarvis"

Say "hey jarvis". Do not expect "Milo" to wake the speaker before this step.

Licence
-------

The pretrained hey_jarvis model is published under CC BY-NC-SA 4.0.
A model you train yourself is the one to use if this speaker is more than
a household prototype. openWakeWord's code is Apache-2.0.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the Milo wake-word training plan.")
    parser.add_argument("--phrase", default="milo", help="Target phrase. 'hey milo' is recommended.")
    parser.add_argument("--model-name", default="", help="File stem. Defaults to the phrase with spaces as underscores.")
    parser.add_argument("--write", type=Path, default=None, help="Also write the plan to this path.")
    args = parser.parse_args()
    phrase = " ".join(args.phrase.split()).strip().lower()
    model_name = args.model_name or phrase.replace(" ", "_")
    text = PLAN.format(notebook=NOTEBOOK, automatic=AUTOMATIC, synthetic=SYNTHETIC, phrase=phrase, model_name=model_name).strip() + "\n"
    print(text)
    if args.write:
        args.write.write_text(text, encoding="utf-8")
    if phrase == "milo":
        print("Note: 'milo' is only two syllables. Train 'hey milo' as well and keep whichever false-wakes less.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
