#!/bin/bash
# Voice Python packages for Raspberry Pi OS (Bookworm or Trixie).
#
# openwakeword depends on tflite-runtime on Linux. That project publishes no
# wheel for Python 3.13, so `pip install openwakeword` fails on current Pi OS
# and the old installer then continued without sounddevice or the wake word.
# Install the libraries it needs, then install openwakeword itself with
# --no-deps. The runtime loads ONNX models only.

install_voice_python() {
  local pip="$1"
  local app="$2"

  "$pip" install --upgrade pip
  if ! "$pip" install -e "$app"; then
    echo "Failed to install the gva package from $app." >&2
    return 1
  fi
  if ! "$pip" install \
    'faster-whisper>=1.0.3' \
    'piper-tts>=1.2.0' \
    'sounddevice>=0.4.6' \
    'onnxruntime>=1.20' \
    'scipy>=1.11' \
    'scikit-learn>=1.3' \
    'tqdm>=4.66' \
    'requests>=2.31'
  then
    echo "Voice packages failed to install. Not continuing without the microphone, speaker, and wake word." >&2
    return 1
  fi
  if ! "$pip" install --no-deps 'openwakeword>=0.6.0'; then
    echo "openwakeword failed to install. Not continuing without the wake word." >&2
    echo "It is installed with --no-deps so pip does not pull tflite-runtime." >&2
    return 1
  fi
}
