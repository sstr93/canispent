"""
Step 3 — Shot list generation.
Calls Claude claude-sonnet-4-20250514 with audio analysis + lyrics and asks it to
write a 20-25 shot list. Saves to shots/treatment.json.
"""

import argparse
import json
import os
from pathlib import Path

import anthropic


_SYSTEM = (
    "You are an experienced music video creative director. "
    "You write precise, imaginative shot lists that sync visually to music. "
    "Your image generation prompts are detailed and production-ready."
)

_USER_TEMPLATE = """\
Write a shot list for a professional music video based on the data below.

=== SONG ANALYSIS ===
BPM: {bpm}
Duration: {duration}s

Sections:
{sections}

Beat timestamps (first 30): {beats_sample}
Total beats: {total_beats}

=== LYRICS ===
{lyrics}

=== INSTRUCTIONS ===
Produce exactly {target_shots} shots that:
- Cover the full song duration from 0s to {duration}s with no gaps
- Sync shot cuts to beat timestamps where possible
- Reflect the mood and content of the lyrics in each shot
- Vary between close-up, medium, wide, and abstract shots
- Tell a coherent visual narrative arc (build → peak → resolve)
- Use cinematic, stylistically consistent image generation prompts

Return ONLY a JSON array — no markdown fences, no commentary:
[
  {{
    "shot_number": 1,
    "start_beat_time": 0.0,
    "end_beat_time": 4.2,
    "section": "intro",
    "prompt": "Highly detailed Flux image generation prompt. Describe subject, environment, lighting, color palette, and artistic style.",
    "visual_mood": "Short mood tag, e.g. 'ethereal and melancholic'",
    "camera_motion": "e.g. 'slow push in', 'pan left to right', 'static wide', 'handheld close-up'",
    "description": "One sentence: what the viewer sees in this shot."
  }}
]"""


def _build_prompt(analysis: dict, lyrics: dict, target_shots: int = 22) -> str:
    sections_text = "\n".join(
        f"  {s['label']:12s} {s['start']:.1f}s – {s['end']:.1f}s"
        for s in analysis["sections"]
    )
    beats = analysis["beat_times"]
    beats_sample = ", ".join(f"{t:.2f}s" for t in beats[:30])
    lyrics_text = lyrics.get("full_text") or "(No lyrics / instrumental)"

    return _USER_TEMPLATE.format(
        bpm=analysis["bpm"],
        duration=analysis["duration"],
        sections=sections_text,
        beats_sample=beats_sample,
        total_beats=len(beats),
        lyrics=lyrics_text,
        target_shots=target_shots,
    )


def generate_treatment(output_dir: str = "shots", target_shots: int = 22) -> list:
    analysis_path = Path(output_dir) / "analysis.json"
    lyrics_path = Path(output_dir) / "lyrics.json"

    if not analysis_path.exists():
        raise FileNotFoundError(f"{analysis_path} not found — run analyze.py first.")
    if not lyrics_path.exists():
        raise FileNotFoundError(f"{lyrics_path} not found — run transcribe.py first.")

    analysis = json.loads(analysis_path.read_text())
    lyrics = json.loads(lyrics_path.read_text())

    print("Calling Claude claude-sonnet-4-20250514 to generate shot list…")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _build_prompt(analysis, lyrics, target_shots)}],
        )
    except anthropic.APIConnectionError:
        raise RuntimeError("Could not reach the Anthropic API. Check your internet connection.")
    except anthropic.AuthenticationError:
        raise RuntimeError("Invalid ANTHROPIC_API_KEY. Check your .env file.")
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}")

    raw = message.content[0].text.strip()

    # Strip markdown fences if the model added them anyway
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        shots = json.loads(raw)
    except json.JSONDecodeError as e:
        print("ERROR: Claude returned invalid JSON.")
        print("Raw response:\n", raw[:1000])
        raise RuntimeError(f"JSON parse error: {e}")

    if not isinstance(shots, list):
        raise RuntimeError("Claude response was not a JSON array.")

    treatment = {
        "song_path": analysis["song_path"],
        "bpm": analysis["bpm"],
        "duration": analysis["duration"],
        "shots": shots,
    }

    out = Path(output_dir) / "treatment.json"
    out.write_text(json.dumps(treatment, indent=2))
    print(f"Saved: {out}  ({len(shots)} shots)")
    return shots


def print_treatment(shots: list) -> None:
    width = 72
    print("\n" + "=" * width)
    print("SHOT LIST / TREATMENT")
    print("=" * width)
    for s in shots:
        header = (
            f"Shot {s['shot_number']:>2}  "
            f"{s['start_beat_time']:.1f}s–{s['end_beat_time']:.1f}s  "
            f"[{s['section']}]"
        )
        print(f"\n{header}")
        print(f"  {s['description']}")
        print(f"  Mood:   {s['visual_mood']}")
        print(f"  Camera: {s['camera_motion']}")
        prompt_preview = s["prompt"][:110] + ("…" if len(s["prompt"]) > 110 else "")
        print(f"  Prompt: {prompt_preview}")
    print("\n" + "=" * width)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate shot list with Claude")
    parser.add_argument("--output-dir", default="shots")
    parser.add_argument("--shots", type=int, default=22, help="Target number of shots")
    args = parser.parse_args()

    shots = generate_treatment(args.output_dir, args.shots)
    print_treatment(shots)
