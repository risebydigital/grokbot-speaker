<p align="center">
  <img src="docs/assets/banner.svg" alt="Milo, a Grok Voice speaker by Rise By Digital" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square" alt="Python 3.11 or newer">
  <img src="https://img.shields.io/badge/Raspberry%20Pi-4-C51A4A?style=flat-square" alt="Raspberry Pi 4">
  <img src="https://img.shields.io/badge/licence-private-052A48?style=flat-square" alt="Private licence">
</p>

<p align="center"><strong>Milo</strong> is a headless Grok voice speaker for a Raspberry Pi 4: say the wake phrase, talk, and hear the reply on the USB speakers.</p>

## What it does

Milo sits in a room, listens for a wake phrase, and holds a spoken conversation with xAI's realtime voice API. Replies play through a USB speaker. A redacted copy of each turn is stored on the Pi and synced to Supabase for the Voice Ops Dashboard.

The default path is speech-to-speech over a WebSocket (`grok-voice-latest`, voice `eve`). A local cascade (faster-whisper, Grok chat, Piper) stays available when you set `conversation.engine` to `local`.

Until a custom model is trained, the wake phrase is **hey jarvis**. The product name is Milo.

## Architecture

```mermaid
flowchart LR
  Mic["AIRHUG microphone"] --> Wake["openWakeWord: Milo"]
  Wake --> Voice["xAI realtime voice"]
  Voice --> Spk["USB speakers"]
  Voice --> Redact["Redacted telemetry"]
  Redact --> Supa["Supabase"]
  Supa --> Dash["Voice Ops Dashboard"]
```

Capture is 48 kHz from the microphone. Wake-word scoring uses 16 kHz. The realtime socket uses 24 kHz PCM. Playback is one long-lived output stream with a short jitter buffer, fed by a queue so each audio delta does not reopen the device.

## Hardware

| Piece | How it is used |
| --- | --- |
| Raspberry Pi 4, 4 GB | Runs the voice loop under systemd. PipeWire is the audio server. |
| AIRHUG USB microphone | Capture. Two inputs, native 48 kHz, no hardware echo cancellation. |
| Sanyun SW208 | Active bookshelf speakers. USB into a Pi USB-A port. The card used here accepts S16_LE, 2 channels, 48000 Hz only. |

The Pi 4 USB-C socket is power only. Do not put the speaker there. Bluetooth is not on the default playback path. If no USB speaker is present, Milo uses the 3.5 mm headphone jack.

## Quick start

On the Pi, over SSH:

```bash
ssh <user>@<pi-hostname>
git clone <repository-url> milo
cd milo
sudo bash deploy/install.sh
```

The installer copies the tree to `/opt/gva/app`, creates a virtualenv at `/opt/gva/venv`, and installs systemd units. It does not start them until `gva doctor` passes.

Edit the secrets file as root, then check the box:

```bash
sudoedit /etc/gva/gva.env
sudo gva doctor
gva audio devices
gva test-audio
sudo systemctl enable --now gva-voice.service gva-backend.service gva-retention.timer gva-backup.timer
```

`gva doctor`, `gva audio devices`, and `gva test-audio` do not need root. `gva doctor` reads `/etc/gva/gva.env` when the file is readable (root, or the `gva` user). It never prints secret values. If the file cannot be read it says so, instead of claiming the keys are missing.

Enable `gva-sync.service` only after `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are set. The installer leaves it disabled until then. A blank URL makes `gva sync run` log that sync is idle and exit 0, so systemd does not restart it in a loop.

`gva doctor` must succeed before the voice service is enabled. A failed doctor run still installs the units, and leaves them stopped.

## Updating

On the Pi the app checkout is owned by `gva`, and neither root nor `gva` has GitHub credentials. Fetch in the operator checkout first, then update:

```bash
cd ~/milo
git pull --ff-only
sudo bash deploy/update.sh
gva doctor
gva test-audio
```

`deploy/update.sh` sets `safe.directory` so git does not refuse the `gva`-owned tree. When `/home/$SUDO_USER/milo` exists it fast-forwards `/opt/gva/app` from that checkout (`git pull --ff-only ... main`). Otherwise it pulls `origin`. It restarts only units that are enabled. Re-running `sudo bash deploy/install.sh` updates an existing checkout the same way, instead of leaving the old tree in place.

## Audio test

`gva test-audio` plays a short tone on the device Milo will use and prints the card name.

```bash
gva test-audio
gva test-audio --seconds 3
gva audio devices
```

An empty `audio.output_device` means: prefer a USB playback device whose name does not contain AIRHUG, otherwise the headphone jack.

## Configuration

Non-secret settings live in `/etc/gva/config.yaml`, seeded from [`config/config.example.yaml`](config/config.example.yaml).

| Key | Default | Meaning |
| --- | --- | --- |
| `conversation.engine` | `realtime` | `realtime` or `local` |
| `conversation.realtime.model` | `grok-voice-latest` | Realtime model |
| `conversation.realtime.voice` | `eve` | Spoken voice |
| `audio.input_device` | `AIRHUG` | Substring of the capture device |
| `audio.output_device` | empty | Empty selects USB, then the 3.5 mm jack |
| `audio.capture_rate` | `48000` | Microphone rate |
| `audio.playback_prebuffer_ms` | `200` | Audio queued before the speaker leaves silence |
| `audio.playback_latency` | `high` | PortAudio latency hint (`high` or `low`) |
| `wake.model` | `hey_jarvis` | openWakeWord model name, or a path to ONNX |
| `wake.phrase` | `hey jarvis` | Phrase you say |
| `wake.threshold` | `0.6` | Score that counts as a hit |
| `wake.patience_frames` | `2` | Hits in a row before the session opens |
| `wake.cooldown_seconds` | `3` | Quiet time after a session |
| `bot_bridge.enabled` | `true` | Off when false. Otherwise on once the webhook and Supabase are set |
| `bot_bridge.timeout_s` | `180` | How long Milo waits for the team |
| `bot_bridge.poll_s` | `3` | How often Milo checks for an answer |
| `bot_bridge.key_header` | `Authorization` | Header name copied from the Grok Bot routine panel |
| `bot_bridge.key_prefix` | `Bearer ` | Prefix copied from that panel, including the trailing space |

Change an allow-listed key with `gva config set <key> <value>`. Secrets are rejected there. They belong only in the env file.

## Bot bridge

Quick chat stays on the xAI realtime session. A Rise By Digital work question (Asana tasks, reports that are due, client work, calendar, email) is handed to Adam's Grok Bot bots, which already have those connections. Milo does not answer that kind of question from the voice session.

The bridge turns on by itself when all four of these are set:

- `GVA_GROKBOT_WEBHOOK_URL`
- `GVA_GROKBOT_WEBHOOK_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

`bot_bridge.enabled: false` forces it off even when those values are present. While it is off, the `ask_rbd_team` tool is not offered to the model.

On a work question Milo makes a `request_id` and posts `{request_id, device_id, question, asked_at, reply_by}` to the webhook. The header name and prefix come from `bot_bridge.key_header` and `bot_bridge.key_prefix`. Copy both from the Grok Bot routine panel, and quote the prefix so a trailing space is kept (`key_prefix: "Bearer "`). The post times out after 10 seconds and is tried once more. The key is never written to the log. A query string on the webhook URL is never written either.

Milo says it is checking with the team, then polls `public.milo_bot_answers` until the answer arrives or `bot_bridge.timeout_s` runs out. Polling continues after the voice session has closed and Milo is back on the wake word. When a row arrives, Milo reads it out. If the session is still open, it is spoken there. If the session has closed, Milo plays a short chime, then reads the text on a short realtime session (verbatim), or with Piper if that session cannot start. The row's `read_at` is filled in afterwards. If nothing arrives in time, Milo says: "Sorry, the team didn't get back to me in time. Try again in a bit."

`bridge.sent`, `bridge.answered` (with `latency_ms`), `bridge.timeout`, and `bridge.error` go to the local event log and the on-device database. They are not uploaded with the voice-ops sync, and they do not contain the question or the answer.

### One-off setup

This repository does not create the table. Add it in the Supabase project (Lovable is a fine place to do that) before turning the bridge on. There is no migration for it here.

`public.milo_bot_answers`

| Column | Type |
| --- | --- |
| `request_id` | text, primary key |
| `device_id` | text |
| `answer` | text, not null |
| `answered_by` | text |
| `created_at` | timestamptz, default `now()` |
| `read_at` | timestamptz |

Row level security on, and only the service role may read or write. An edge function named `milo-answer` is what writes the rows when a bot replies.

On the Pi, after this branch is on `main`:

```bash
cd ~/milo
git pull --ff-only
sudo bash deploy/update.sh
```

Then:

1. Put the webhook URL and key in `/etc/gva/gva.env` (mode `0600`, owner `gva`). The names are in `gva.env.example`. Leave both blank to keep the bridge off.
2. In `/etc/gva/config.yaml`, set `bot_bridge.key_header` and `bot_bridge.key_prefix` to the format from the Grok Bot routine panel.
3. Run `gva doctor`. It prints `bridge: configured, milo_bot_answers reachable`, or `bridge: not configured`. It does not print the key or the webhook URL. If the table cannot be reached, that is a warning. The voice service still starts.
4. Run `gva bridge test "What reports are due this week?"` and wait for the answer.
5. If `gva-voice` is already enabled, `deploy/update.sh` restarts it. Otherwise `sudo systemctl restart gva-voice`.

## Troubleshooting

### Choppy playback

Realtime audio arrives as short 24 kHz mono deltas. The Sanyun card only accepts 48 kHz stereo, so each delta is resampled. Milo keeps one callback stream open for the whole reply (`latency` `high` by default) and resamples with a filter that remembers its state, so the joins do not click. Playback waits until about 200 ms is buffered (`audio.playback_prebuffer_ms`). If the buffer runs dry the callback writes silence and counts an underrun. It does not close and reopen the device. `playback.completed` records `audio_ms` and `underruns`. `stt.completed` records `audio_ms` and `duration_ms`.

If a reply is still uneven:

1. Run `gva test-audio`. A broken tone means the card or the cable, not the voice socket.
2. Confirm the speaker is on a USB-A port, not the power socket.
3. Run `gva audio devices` and check the chosen index.
4. On PipeWire, confirm the USB card is the one that actually receives playback.

### Wake phrase fires twice

After every session Milo drops the openWakeWord model and loads a fresh one, requires two frames over the threshold, flushes the capture queue, and waits out `wake.cooldown_seconds` (3 seconds). That is what stopped the phrase from tripping again on its own echo. Do not lower patience to 1 on this microphone.

### Python 3.13 and the wake word

Raspberry Pi OS Trixie ships Python 3.13. `openwakeword` depends on `tflite-runtime` on Linux, and that project has no wheel for 3.13, so `pip install -e ".[voice]"` fails. `deploy/install.sh` does not ignore that failure. It installs faster-whisper, piper-tts, sounddevice, onnxruntime, scipy, scikit-learn, tqdm, and requests, then installs `openwakeword` with `--no-deps`. Milo loads ONNX models only (`inference_framework="onnx"`), including the melspectrogram and embedding nets.

`gva doctor` reports any of those imports that are missing. `gva models download` fetches the ONNX files, not the TFLite ones.

### Speaker format

`gva test-audio` used to open the output stream at the source rate in mono (16 kHz for a local tone, 24 kHz for realtime). The Sanyun card rejects that with PortAudio error -9997 (invalid sample rate). The sink now asks the device what it will accept: the source rate first, then `default_samplerate` (48000 on this speaker), mono first and stereo if mono is refused. Playback is resampled with state carried across chunks and, when needed, duplicated to both channels before it enters the jitter buffer, so the buffer matches the open stream.

The microphone path does the same in reverse. The AIRHUG is opened at a rate and channel count it supports (native 48 kHz) and the capture callback downmixes to mono and resamples to the rate the pipeline asked for.

A tone from `speaker-test -D plughw:<card>,<device>` means the hardware is fine. If `gva test-audio` still fails, the line it prints (`device ... Hz, ... ch`) is the format it actually opened.

### Config file permissions

`/etc/gva/config.yaml` has no secrets. It is mode `0644`, and `/etc/gva` is mode `0755`, so a normal login can run `gva test-audio`. `/etc/gva/gva.env` stays mode `0600`, owner `gva`. If a config file cannot be read, the command says so and continues with the bundled example instead of raising `PermissionError`.

Logs and the database go to `/var/lib/gva` when that directory is writable, and otherwise to `~/.local/state/gva`. They are never created inside the application directory. `/usr/share/gva/wake`, `voices`, and `fallback` are mode `0755` so the models can be read. Logs and the SQLite database stay private to `gva` when the service is running.

### Headless ALSA

`gva-voice.service` runs as user `gva`, group `gva`, with supplementary group `audio`. That is what lets it open `/dev/snd` without a logged-in seat. There is no PipeWire or PulseAudio session for that user, and the service does not use one. PortAudio opens the ALSA hw device (the name looks like `USB2.0 Device: Audio (hw:4,0)`). PipeWire and Pulse device names are skipped so a desktop session graph is not selected by mistake.

`PrivateDevices` is left unset, so `/dev/snd` is visible inside the sandbox. `ProtectSystem=strict` does not hide device nodes. If a logged-in PipeWire process already holds the USB card, the service gets "device or resource busy". On a headless Pi nothing else grabs the card. Check with:

```bash
aplay -l
speaker-test -D plughw:<card>,<device> -c 2 -r 48000 -f 440 -t sine -l 1
gva audio devices
gva test-audio
```

`gva audio devices` does not read `/etc/gva/config.yaml`.

### Finding the audio device

```bash
gva audio devices
aplay -l
```

Match a substring into `audio.input_device` or `audio.output_device`. Leave the output blank to keep the USB-then-jack rule. The AIRHUG name is ignored for playback even if it appears in the list.

## Security

API keys and the Supabase service-role key live only in `/etc/gva/gva.env` on the Pi. Root creates that file. Mode is `0600`, owner `gva`, so the voice service can read it and other accounts cannot. It is never committed, never logged, and never sent to the dashboard.

The dashboard reads Supabase with the anon key under row-level security. The service-role key stays on the Pi. Local logs pass through the redaction pipeline before sync. Do not put household secrets in `config.yaml`.

## Roadmap

- A trained Milo wake model, via `tools/wake/train_milo.py`. Until that file is installed, the phrase stays "hey jarvis".
- The Voice Ops Dashboard, reading redacted telemetry from Supabase.
- The local whisper / Piper cascade remains the fallback when the realtime socket should stay closed.

## Credits

The loop is original code. These projects shaped it, and their sources were not copied:

- [m15-ai/Grok-Voice](https://github.com/m15-ai/Grok-Voice) - realtime xAI voice on a Pi, USB audio, and PipeWire.
- [herrkaefer/snowman](https://github.com/herrkaefer/snowman) - a fresh wake decoder for each listening period, light echo suppression, and `response.cancel` when the wake phrase barges in.
- [ErwanHeschung/Naka](https://github.com/ErwanHeschung/Naka) - playback pulled from an `asyncio` queue, plus a watchdog on a stalled socket. Naka is MIT; the idea was reimplemented here.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check gva tests tools
pytest
python tools/secret_scan.py
```

On the Pi, `deploy/install.sh` installs the voice packages. Do not `pip install openwakeword` on its own: on Python 3.13 that pulls `tflite-runtime`, which has no wheel. Tests do not call xAI and do not need a microphone.
