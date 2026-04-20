"""
Step 1 — Audio analysis.
Extracts BPM, beat timestamps, section boundaries, and energy curve.
Saves results to shots/analysis.json.
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np


def analyze_audio(song_path: str, output_dir: str = "shots") -> dict:
    print(f"Loading audio: {song_path}")
    y, sr = librosa.load(str(song_path))
    duration = librosa.get_duration(y=y, sr=sr)

    # BPM and beat grid
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    bpm = float(tempo)

    # Section boundaries via MFCC + agglomerative clustering
    hop_length = 512
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    n_sections = min(8, max(2, int(duration / 15)))  # ~15s per section, 2–8 sections
    bounds = librosa.segment.agglomerative(mfcc, k=n_sections)
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop_length).tolist()

    # Ensure we always start at 0 and end at duration
    if bound_times[0] > 0.5:
        bound_times.insert(0, 0.0)
    if bound_times[-1] < duration - 0.5:
        bound_times.append(duration)

    # Label sections heuristically by position in song
    labels = _heuristic_labels(len(bound_times) - 1)
    sections = [
        {
            "label": labels[i],
            "start": round(bound_times[i], 3),
            "end": round(bound_times[i + 1], 3),
            "duration": round(bound_times[i + 1] - bound_times[i], 3),
        }
        for i in range(len(bound_times) - 1)
    ]

    # Per-second RMS energy curve
    frame_len = sr  # 1-second frames
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=frame_len)[0]

    analysis = {
        "song_path": str(song_path),
        "duration": round(duration, 3),
        "bpm": round(bpm, 2),
        "beat_times": [round(t, 3) for t in beat_times],
        "sections": sections,
        "energy_curve": [round(float(e), 4) for e in rms],
        "sample_rate": sr,
    }

    out = Path(output_dir) / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2))

    print(f"  BPM:      {bpm:.1f}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Beats:    {len(beat_times)}")
    print(f"  Sections: {len(sections)} → {[s['label'] for s in sections]}")
    print(f"Saved: {out}")
    return analysis


def _heuristic_labels(n: int) -> list[str]:
    """Map section count to song-structure labels."""
    templates = {
        2: ["verse", "chorus"],
        3: ["verse", "chorus", "outro"],
        4: ["intro", "verse", "chorus", "outro"],
        5: ["intro", "verse", "chorus", "verse", "outro"],
        6: ["intro", "verse", "chorus", "verse", "chorus", "outro"],
        7: ["intro", "verse", "pre-chorus", "chorus", "verse", "chorus", "outro"],
        8: ["intro", "verse", "pre-chorus", "chorus", "verse", "chorus", "bridge", "outro"],
    }
    template = templates.get(n, [f"section_{i}" for i in range(n)])
    # Pad or trim to match n
    while len(template) < n:
        template.append(f"section_{len(template)}")
    return template[:n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze audio file")
    parser.add_argument("song", help="Path to the song file")
    parser.add_argument("--output-dir", default="shots")
    args = parser.parse_args()
    analyze_audio(args.song, args.output_dir)
