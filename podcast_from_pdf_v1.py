#!/usr/bin/env python3
"""
Local NotebookLM clone — turn a PDF into a two-host podcast.
Runs entirely on-device using MLX on Apple Silicon.

Usage:
    python podcast_from_pdf.py path/to/paper.pdf

Setup (once):
    pip install mlx-lm mlx-audio pypdf soundfile numpy

Outputs land in ./podcast_output/
    - script.txt        (the generated dialogue)
    - clips/*.wav       (per-line audio)
    - podcast.wav       (final stitched episode)
"""

import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pypdf import PdfReader

from mlx_lm import load, generate
from mlx_audio.tts.generate import generate_audio

# ---------- Config ----------
LLM_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
TTS_MODEL = "prince-canuma/Kokoro-82M"

# Two distinct Kokoro voices. Swap to taste — there are ~20 to choose from.
VOICE_A = "af_heart"     # Alex — curious host
VOICE_B = "am_michael"   # Sam — expert host

OUTPUT_DIR = Path("podcast_output")
MAX_DOC_CHARS = 20_000   # truncate huge PDFs to fit the LLM context
GAP_MS = 250             # silence between lines

SCRIPT_PROMPT = """You are a podcast script writer. Convert the document below into an engaging ~5-minute podcast conversation between two hosts:

- HOST_A is "Alex" — curious, asks great questions, occasionally reacts ("hmm", "wait — really?")
- HOST_B is "Sam" — the expert. Explains clearly, uses vivid analogies, never lectures.

Strict output rules:
- Output ONLY the script, no preamble, no markdown, no stage directions.
- Every line must start with "HOST_A:" or "HOST_B:" followed by what that host says aloud.
- Keep each line under 40 words so it sounds natural when spoken.
- Aim for 30–50 turns total.
- Open with Alex welcoming listeners and teeing up the topic.
- Close with a memorable one-line takeaway from Sam.
- Make it a real conversation — interruptions, follow-ups, mild disagreement are great.

DOCUMENT:
---
{doc}
---
Now write the script."""


def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    text = re.sub(r"\s+\n", "\n", text).strip()
    return text[:MAX_DOC_CHARS]


def generate_script(doc_text: str) -> str:
    print("[1/3] Loading LLM and generating script...")
    model, tokenizer = load(LLM_MODEL)

    messages = [{"role": "user", "content": SCRIPT_PROMPT.format(doc=doc_text)}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    script = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=4000,
        verbose=True,  # streams tokens to terminal — looks great live
    )
    return script


def parse_script(script: str):
    """Parse LLM output into [(voice_id, text), ...]."""
    lines = []
    pattern = re.compile(r"^\s*HOST_([AB])\s*:\s*(.+?)\s*$", re.IGNORECASE)
    for raw in script.splitlines():
        m = pattern.match(raw)
        if not m:
            continue
        host, text = m.group(1).upper(), m.group(2)
        if not text:
            continue
        voice = VOICE_A if host == "A" else VOICE_B
        lines.append((voice, text))
    return lines


def synthesize_lines(lines, clip_dir: Path):
    print(f"[2/3] Synthesizing {len(lines)} lines with Kokoro...")
    clip_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, (voice, text) in enumerate(lines):
        prefix = clip_dir / f"line_{i:03d}"
        generate_audio(
            text=text,
            model_path=TTS_MODEL,
            voice=voice,
            speed=1.0,
            file_prefix=str(prefix),
            audio_format="wav",
            sample_rate=24000,
            join_audio=True,
            verbose=False,
        )
        # mlx-audio appends suffixes sometimes — grab whatever it produced
        produced = sorted(clip_dir.glob(f"line_{i:03d}*.wav"))
        if not produced:
            print(f"  !! no audio produced for line {i}, skipping")
            continue
        files.append(produced[0])
        print(f"  {i+1}/{len(lines)}  [{('Alex' if voice == VOICE_A else 'Sam'):4s}]  {text[:60]}")
    return files


def stitch(audio_files, output_path: Path):
    print("[3/3] Stitching final episode...")
    parts = []
    sr = None
    for f in audio_files:
        audio, file_sr = sf.read(str(f))
        if sr is None:
            sr = file_sr
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono
        parts.append(audio.astype(np.float32))
        parts.append(np.zeros(int(sr * GAP_MS / 1000), dtype=np.float32))
    final = np.concatenate(parts)
    # Light normalization so it sounds polished
    peak = np.max(np.abs(final)) or 1.0
    final = (final / peak) * 0.9
    sf.write(str(output_path), final, sr)
    duration = len(final) / sr
    print(f"\n✅ Saved {output_path}  ({duration:.1f}s, {duration/60:.1f} min)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python podcast_from_pdf.py <pdf_path>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    doc = extract_pdf(pdf_path)
    print(f"Extracted {len(doc):,} chars from {pdf_path.name}")

    script = generate_script(doc)
    (OUTPUT_DIR / "script.txt").write_text(script)

    lines = parse_script(script)
    print(f"Parsed {len(lines)} dialogue lines\n")
    if not lines:
        print("⚠️  LLM didn't follow the HOST_A/HOST_B format. Inspect script.txt.")
        sys.exit(1)

    clips = synthesize_lines(lines, OUTPUT_DIR / "clips")
    if not clips:
        print("No audio clips produced — check TTS model and voice names.")
        sys.exit(1)
    stitch(clips, OUTPUT_DIR / "podcast.wav")


if __name__ == "__main__":
    main()
