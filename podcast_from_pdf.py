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
VOICE_A = "af_sarah"     # Alex — warm, curious host
VOICE_B = "bm_george"    # Sam — authoritative, expert host

OUTPUT_DIR = Path("podcast_output")
MAX_DOC_CHARS = 20_000   # truncate huge PDFs to fit the LLM context

# Per-emotion profile: (tts_speed, pitch_semitones, gain, gap_after_ms)
# pitch_semitones — applied via numpy resampling after synthesis, no extra deps.
#   +N = higher/brighter (excited), -N = lower/darker (thoughtful).
# gain — volume multiplier applied before stitching.
# gap_after_ms — silence inserted after this line; long pauses feel dramatic.
EMOTION_PROFILE: dict[str, tuple[float, float, float, int]] = {
    "excited":    (1.00,  +2.5,  1.08,  100),
    "curious":    (0.92,  +0.5,  1.00,  220),
    "surprised":  (1.18,  +3.0,  1.06,  140),
    "emphatic":   (0.95,   0.0,  1.15,  200),
    "thoughtful": (0.76,  -2.0,  0.92,  420),
    "warm":       (0.84,  -1.2,  0.94,  310),
    "neutral":    (1.00,   0.0,  1.00,  250),
}
_DEFAULT_PROFILE = (1.00, 0.0, 1.00, 250)

SCRIPT_PROMPT = """You are a podcast script writer. Convert the document below into an engaging ~5-minute podcast conversation between two hosts:

- HOST_A is "Alex" — curious science journalist, 8 years covering emerging tech. Asks great questions, occasionally reacts ("hmm", "wait — really?")
- HOST_B is "Sam" — senior research scientist with 12 years in the field. Explains clearly, uses vivid analogies, never lectures.

Strict output rules:
- Output ONLY the script, no preamble, no markdown, no stage directions.
- Every line must start with "HOST_A:" or "HOST_B:", then an emotion tag in square brackets, then what that host says aloud.
- Valid emotion tags: [excited] [curious] [surprised] [emphatic] [thoughtful] [warm] [neutral]
- Choose the tag that best fits the line's tone. Use [excited] and [surprised] sparingly.
- Within the spoken text, use natural punctuation for rhythm: commas for micro-pauses, "..." for longer pauses, "!" for genuine energy. Capitalise a word for spoken stress (e.g. "that is WILD").
- Keep each line under 40 words so it sounds natural when spoken.
- Aim for 30–50 turns total.
- FIRST LINE RULE: The very first line each host speaks must begin with a natural one-liner that mentions their name and relevant background before continuing — woven in conversationally, not as a formal announcement. This happens ONLY on each host's first turn; all subsequent turns jump straight into the dialogue.
- Open with Alex welcoming listeners, introducing himself, and teeing up the topic.
- Sam's first line must naturally drop in his name and background before engaging with the topic.
- Close with a memorable one-line takeaway from Sam.
- Make it a real conversation — interruptions, follow-ups, mild disagreement are great.

Example first lines:
HOST_A: [warm] Hey everyone, I'm Alex — I've been covering emerging tech for eight years now — and today we're diving into something that genuinely blew my mind.
HOST_B: [warm] Thanks Alex — I'm Sam, I've spent the last twelve years in the lab working on exactly this kind of problem, so... yeah, I have a few thoughts.

Example subsequent lines (NO intro):
HOST_A: [curious] So... what actually makes this approach different from everything that came before?
HOST_B: [emphatic] The key insight is that it's not about speed — it's about PRECISION at scale.

DOCUMENT:
---
{doc}
---
Now write the script."""


def pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Resample to shift pitch (also slightly shortens/lengthens duration — natural for emotion)."""
    if semitones == 0.0:
        return audio
    factor = 2.0 ** (semitones / 12.0)
    new_len = max(1, int(round(len(audio) / factor)))
    return np.interp(
        np.linspace(0, len(audio) - 1, new_len),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


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
    """Parse LLM output into [(voice_id, emotion, text), ...]."""
    lines = []
    # Match:  HOST_A: [emotion] spoken text
    pattern = re.compile(
        r"^\s*HOST_([AB])\s*:\s*\[(\w+)\]\s*(.+?)\s*$", re.IGNORECASE
    )
    # Fallback without emotion tag
    fallback = re.compile(r"^\s*HOST_([AB])\s*:\s*(.+?)\s*$", re.IGNORECASE)
    for raw in script.splitlines():
        m = pattern.match(raw)
        if m:
            host, emotion, text = m.group(1).upper(), m.group(2).lower(), m.group(3)
        else:
            m = fallback.match(raw)
            if not m:
                continue
            host, emotion, text = m.group(1).upper(), "neutral", m.group(2)
        if not text:
            continue
        voice = VOICE_A if host == "A" else VOICE_B
        lines.append((voice, emotion, text))
    return lines


def synthesize_lines(lines, clip_dir: Path):
    """Returns list of (wav_path, locked_pitch, gain, gap_ms) tuples."""
    print(f"[2/3] Synthesizing {len(lines)} lines with Kokoro...")
    clip_dir.mkdir(parents=True, exist_ok=True)
    # Speed and pitch are locked per host after their first line.
    # Gain and gap still vary by emotion for expressiveness.
    host_speed: dict[str, float] = {}
    host_pitch: dict[str, float] = {}
    files = []
    for i, (voice, emotion, text) in enumerate(lines):
        tts_speed, semitones, gain, gap_ms = EMOTION_PROFILE.get(emotion, _DEFAULT_PROFILE)
        if voice not in host_speed:
            host_speed[voice] = tts_speed
            host_pitch[voice] = semitones
        tts_speed = host_speed[voice]
        semitones  = host_pitch[voice]
        prefix = clip_dir / f"line_{i:03d}"
        generate_audio(
            text=text,
            model_path=TTS_MODEL,
            voice=voice,
            speed=tts_speed,
            file_prefix=str(prefix),
            audio_format="wav",
            sample_rate=24000,
            join_audio=True,
            verbose=False,
        )
        produced = sorted(clip_dir.glob(f"line_{i:03d}*.wav"))
        if not produced:
            print(f"  !! no audio produced for line {i}, skipping")
            continue
        files.append((produced[0], semitones, gain, gap_ms))
        name = "Alex" if voice == VOICE_A else "Sam"
        print(f"  {i+1}/{len(lines)}  [{name:4s}][{emotion:10s}] spd={tts_speed:.2f} pitch={semitones:+.1f}  {text[:45]}")
    return files


def stitch(audio_files, output_path: Path):
    """audio_files: list of (path, locked_pitch, gain, gap_ms)."""
    print("[3/3] Stitching final episode...")
    parts = []
    sr = None
    for f, semitones, gain, gap_ms in audio_files:
        audio, file_sr = sf.read(str(f))
        if sr is None:
            sr = file_sr
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        audio = pitch_shift(audio, semitones)
        audio = audio * gain

        parts.append(audio)
        parts.append(np.zeros(int(sr * gap_ms / 1000), dtype=np.float32))

    final = np.concatenate(parts)
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
    stitch(clips, OUTPUT_DIR / "podcast.wav")  # clips = [(path, emotion, gap_ms), ...]


if __name__ == "__main__":
    main()
