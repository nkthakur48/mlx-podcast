# mlx-podcast

Turn any PDF into a two-host podcast — runs entirely on-device using MLX on Apple Silicon. No API keys, no cloud, no cost per run.

## How it works

```
PDF → extract text → LLM script → parse dialogue → TTS synthesis → stitch audio
```

1. **Extract** — pulls text from the PDF (truncated to 20k chars to fit LLM context)
2. **Script** — Qwen2.5-7B generates a 30–50 turn dialogue between two hosts
3. **Synthesize** — Kokoro-82M TTS renders each line to a `.wav` clip with per-emotion speed/pitch/gain
4. **Stitch** — clips are pitch-shifted, normalized, and concatenated into a single episode

## Hosts

| Host | Voice | Persona |
|------|-------|---------|
| Alex | `af_sarah` | Curious science journalist, 8 years covering emerging tech |
| Sam  | `bm_george` | Senior research scientist, 12 years in the field |

## Emotion system

Each line of dialogue gets an emotion tag that shapes the audio:

| Emotion | Speed | Pitch | Gain | Gap after |
|---------|-------|-------|------|-----------|
| `excited` | 1.00× | +2.5 st | 1.08× | 100 ms |
| `curious` | 0.92× | +0.5 st | 1.00× | 220 ms |
| `surprised` | 1.18× | +3.0 st | 1.06× | 140 ms |
| `emphatic` | 0.95× | 0.0 st | 1.15× | 200 ms |
| `thoughtful` | 0.76× | −2.0 st | 0.92× | 420 ms |
| `warm` | 0.84× | −1.2 st | 0.94× | 310 ms |
| `neutral` | 1.00× | 0.0 st | 1.00× | 250 ms |

Speed and pitch are locked to each host's first-line emotion for consistency; gain and gap vary per line for expressiveness.

## Setup

```bash
pip install mlx-lm mlx-audio pypdf soundfile numpy
```

Requires an Apple Silicon Mac (M1 or later). Models are downloaded automatically on first run.

## Usage

```bash
python podcast_from_pdf.py path/to/paper.pdf
```

## Output

```
podcast_output/
├── script.txt        # generated dialogue
├── clips/
│   ├── line_000.wav  # per-line audio
│   ├── line_001.wav
│   └── ...
└── podcast.wav       # final stitched episode
```

## Models

| Component | Model |
|-----------|-------|
| LLM | [`mlx-community/Qwen2.5-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) |
| TTS | [`prince-canuma/Kokoro-82M`](https://huggingface.co/prince-canuma/Kokoro-82M) |

Kokoro supports ~20 voices — swap `VOICE_A` and `VOICE_B` in the script to change hosts.
