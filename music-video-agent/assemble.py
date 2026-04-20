"""
Step 6 — Final assembly.
Cuts clips together on beat timestamps with crossfades, lays the original
audio over the top, and exports output/final.mp4.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def assemble_video(
    output_dir: str = "shots",
    song_path: str | None = None,
    output_path: str = "output/final.mp4",
    crossfade: float = 0.3,
) -> str:
    # Lazy import — moviepy is slow to import and not needed in dry-run
    try:
        from moviepy.editor import (
            AudioFileClip,
            VideoFileClip,
            concatenate_videoclips,
        )
    except ImportError:
        raise ImportError("moviepy is not installed. Run: pip install 'moviepy<2.0'")

    treatment_path = Path(output_dir) / "treatment.json"
    if not treatment_path.exists():
        raise FileNotFoundError(f"{treatment_path} not found — run treatment.py first.")

    treatment = json.loads(treatment_path.read_text())
    if song_path is None:
        song_path = treatment.get("song_path")
    if not song_path or not Path(song_path).exists():
        raise FileNotFoundError(
            f"Song file not found: {song_path}. Pass --song explicitly."
        )

    shots = treatment["shots"]
    song_duration = treatment["duration"]
    clips_dir = Path(output_dir) / "clips"

    print(f"Assembling {len(shots)} shots…")
    print(f"Song: {song_path} ({song_duration:.1f}s)\n")

    audio = AudioFileClip(str(song_path))

    video_clips = []
    for shot in shots:
        n = shot["shot_number"]
        clip_path = clips_dir / f"shot_{n:03d}.mp4"

        if not clip_path.exists():
            print(f"  Shot {n:>3}: MISSING — skipping ({clip_path.name})")
            continue

        target_dur = shot["end_beat_time"] - shot["start_beat_time"]
        target_dur = max(target_dur, 0.5)  # guard against zero-length slots

        try:
            raw = VideoFileClip(str(clip_path))
        except Exception as e:
            print(f"  Shot {n:>3}: load error — {e}")
            continue

        # Trim or loop to fill the target duration
        if raw.duration >= target_dur:
            clip = raw.subclip(0, target_dur)
        else:
            loops = int(np.ceil(target_dur / raw.duration))
            from moviepy.editor import concatenate_videoclips as _cat

            clip = _cat([raw] * loops).subclip(0, target_dur)

        # Apply crossfade transition (skip on the very first clip)
        if video_clips:
            clip = clip.crossfadein(crossfade)

        video_clips.append(clip)
        print(f"  Shot {n:>3}: {target_dur:.2f}s  ({clip_path.name})")

    if not video_clips:
        raise RuntimeError("No clips could be loaded. Check shots/clips/ directory.")

    print(f"\nConcatenating {len(video_clips)} clips with {crossfade}s crossfades…")
    final = concatenate_videoclips(
        video_clips, method="compose", padding=-crossfade
    )

    # Trim to song length
    final_dur = min(final.duration, song_duration)
    final = final.subclip(0, final_dur)

    # Attach original audio
    final = final.set_audio(audio.subclip(0, final_dur))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting → {out}  (this may take a few minutes)…")
    final.write_videofile(
        str(out),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(out.parent / "temp_audio.m4a"),
        remove_temp=True,
        logger="bar",
    )

    # Release file handles
    final.close()
    audio.close()
    for c in video_clips:
        c.close()

    print(f"\nDone! Final video: {out.absolute()}")
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble clips into final music video")
    parser.add_argument("--output-dir", default="shots")
    parser.add_argument("--song", help="Path to song (overrides value in treatment.json)")
    parser.add_argument("--output", default="output/final.mp4", help="Output file path")
    parser.add_argument("--crossfade", type=float, default=0.3, help="Crossfade duration in seconds")
    args = parser.parse_args()

    assemble_video(args.output_dir, args.song, args.output, args.crossfade)
