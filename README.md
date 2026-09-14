# DiaScribe

**Speaker-aware transcription from the command line.**

Reason I wrote this program? FUN 😁
DiaScribe turns a raw audio file into a clean, speaker-labeled transcript by combining OpenAI's diarization model with Whisper's transcription — reconciling both into a single, readable timeline.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-openai-lightgrey)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## Why

Speech-to-text APIs generally force a tradeoff: a diarization model tells you *who* spoke and *when*, while a dedicated transcription model gives you the most accurate *what*. Few endpoints give you both at full quality in one pass.

DiaScribe doesn't pick a side — it runs both and merges them:

1. **Diarize** — `gpt-4o-transcribe-diarize` segments the audio by speaker turn.
2. **Transcribe** — `whisper-1` produces high-fidelity, timestamped text segments.
3. **Merge** — each transcribed segment is matched to the diarized segment it overlaps with most, assigning it a speaker.
4. **Output** — a single, ordered, speaker-labeled transcript.

## Features

- **Two-model pipeline** — speaker attribution and transcription quality aren't traded off against each other
- **Overlap-based merge** — segments are matched by actual time overlap, not naive index alignment
- **Any OpenAI-compatible endpoint** — point `--base-url` at OpenAI directly or at a compatible proxy/regional provider
- **Flexible output** — stream to stdout or write straight to a file
- **Explicit, categorized error handling** — connection, timeout, rate-limit, and API-status failures are each reported distinctly
- **Small surface area** — one file, one external dependency, no configuration to maintain

## Requirements

- Python 3.9+
- [`openai`](https://pypi.org/project/openai/) Python SDK
- An audio file **≤ 25 MiB**

## Installation

```sh
git clone https://github.com/wizrd00/DiaScribe.git
cd DiaScribe
pip install openai
```

## Usage

```sh
python diascribe.py --file <audio_file> --api-key <API_KEY> --base-url <BASE_URL> [--output <output_file>]
```

| Flag | Required | Description |
|---|---|---|
| `--file` | ✅ | Path to the input audio file (≤ 25 MiB) |
| `--api-key` | ✅ | API key for the OpenAI-compatible endpoint |
| `--base-url` | ✅ | Base URL of the API endpoint |
| `--output` | — | Path to write the transcript to (defaults to stdout) |

### Example

```sh
python diascribe.py --file voice.wav --api-key API_KEY --base-url BASE_URL
```

### Output format

```
[0.00->3.42] Speaker A: Hi, thanks for calling support.
[3.50->6.10] Speaker B: Hey, I'm having trouble with my recent order.
```

Each line is `[start->end] Speaker <label>: <text>`, ordered by the transcription timeline.

## Limitations

- File size is capped at 25 MiB — this mirrors the API's own limit and isn't currently chunked around
- Every run makes two API calls (diarization + transcription), so cost and latency are roughly double a single-model call
- Speaker attribution is a best-effort overlap heuristic

## Possible next steps

- Chunked handling for files over the 25 MiB limit
