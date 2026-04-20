#!/usr/bin/env python3
"""
Music Video Agent — main pipeline orchestrator.

Usage:
  python run.py --song path/to/song.mp3
  python run.py --song path/to/song.mp3 --dry-run
  python run.py --song path/to/song.mp3 --start-from frames
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SHOTS_DIR = "shots"
STEPS = ["analyze", "transcribe", "treatment", "frames", "clips", "assemble"]


# ── Checkpoint helper ────────────────────────────────────────────────────────

def checkpoint(title: str, detail: str = "") -> None:
    """
    Print a checkpoint banner and block until the user approves.
    Typing 'q' or 'quit' exits the whole pipeline cleanly.
    Raises SystemExit on 'q'; returns normally on 'y'.
    """
    width = 72
    print("\n" + "=" * width)
    print(f"  CHECKPOINT: {title}")
    print("=" * width)
    if detail:
        print(detail)
    print()

    while True:
        try:
            answer = input("  Continue? [Y]es / [q]uit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted — exiting.")
            sys.exit(0)

        if answer in ("y", "yes", ""):
            print()
            return
        if answer in ("q", "quit", "exit", "n", "no"):
            print("\nPipeline stopped at your request.")
            print(f"Re-run with --start-from {title.lower().split()[0]} to resume here.")
            sys.exit(0)
        print("  Please type y to continue or q to quit.")


# ── Environment check ────────────────────────────────────────────────────────

def check_env() -> None:
    missing = [k for k in ("ANTHROPIC_API_KEY", "FAL_KEY") if not os.environ.get(k)]
    if missing:
        print("ERROR: Missing required environment variables:")
        for k in missing:
            print(f"  {k}")
        print("\nCopy .env.example to .env and fill in your API keys.")
        sys.exit(1)


# ── Individual step runners ──────────────────────────────────────────────────

def run_analyze(song_path: str) -> dict:
    from analyze import analyze_audio
    return analyze_audio(song_path, SHOTS_DIR)


def run_transcribe(song_path: str) -> dict:
    from transcribe import transcribe_audio
    return transcribe_audio(song_path, SHOTS_DIR)


def run_treatment() -> list:
    from treatment import generate_treatment, print_treatment
    shots = generate_treatment(SHOTS_DIR)
    print_treatment(shots)
    return shots


def run_generate_frames() -> dict:
    from generate_frames import generate_all_frames
    return generate_all_frames(SHOTS_DIR)


def run_generate_clips() -> dict:
    from generate_clips import generate_all_clips
    return generate_all_clips(SHOTS_DIR)


def run_assemble(song_path: str) -> str:
    from assemble import assemble_video
    return assemble_video(SHOTS_DIR, song_path)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _open_folder(path: Path) -> None:
    """Best-effort: open a folder in the system file manager."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=False)
        elif sys.platform == "win32":
            subprocess.run(["explorer", str(path)], check=False)
    except Exception:
        pass


def _prompt_regen_clips(manifest: dict) -> None:
    """Ask the user which failed clips to regenerate before assembling."""
    from generate_clips import regenerate_clips

    failed = manifest.get("failed", [])
    if not failed:
        return

    print(f"\n  Failed clips: {failed}")
    try:
        raw = input(
            "  Enter shot numbers to regenerate (space-separated), or press Enter to skip: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return

    if raw:
        try:
            shot_nums = [int(x) for x in raw.split()]
        except ValueError:
            print("  Could not parse shot numbers — skipping regeneration.")
            return
        regenerate_clips(shot_nums, SHOTS_DIR)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Music Video Agent — AI-powered music video pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python run.py --song song.mp3

  # Test analysis + transcription with no API costs
  python run.py --song song.mp3 --dry-run

  # Resume after already generating frames
  python run.py --song song.mp3 --start-from clips
        """,
    )
    parser.add_argument("--song", required=True, metavar="PATH", help="Path to input audio file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only analysis + transcription (no API costs, no checkpoints)",
    )
    parser.add_argument(
        "--start-from",
        choices=STEPS,
        default="analyze",
        metavar="STEP",
        help=f"Resume from a specific step: {', '.join(STEPS)}",
    )
    args = parser.parse_args()

    song_path = str(Path(args.song).resolve())
    if not Path(song_path).exists():
        print(f"ERROR: Song file not found: {song_path}")
        sys.exit(1)

    if not args.dry_run:
        check_env()

    Path(SHOTS_DIR).mkdir(exist_ok=True)

    width = 72
    print("\n" + "=" * width)
    print("  MUSIC VIDEO AGENT")
    print("=" * width)
    print(f"  Song:    {song_path}")
    print(f"  Output:  {SHOTS_DIR}/ → output/final.mp4")
    if args.dry_run:
        print("  Mode:    DRY RUN (analysis + transcription only, no API costs)")
    if args.start_from != "analyze":
        print(f"  Resuming from: {args.start_from}")
    print("=" * width + "\n")

    start_idx = STEPS.index(args.start_from)

    # ── Step 1: Analyze ──────────────────────────────────────────────────────
    if start_idx <= STEPS.index("analyze"):
        print("[Step 1/6] Analyzing audio…")
        try:
            run_analyze(song_path)
        except Exception as e:
            print(f"ERROR in analyze: {e}")
            sys.exit(1)

    # ── Step 2: Transcribe ───────────────────────────────────────────────────
    if start_idx <= STEPS.index("transcribe"):
        print("\n[Step 2/6] Transcribing lyrics with Whisper…")
        try:
            run_transcribe(song_path)
        except Exception as e:
            print(f"ERROR in transcribe: {e}")
            sys.exit(1)

    if args.dry_run:
        print("\n" + "=" * width)
        print("  DRY RUN COMPLETE")
        print("=" * width)
        print("  Analysis:      shots/analysis.json")
        print("  Transcription: shots/lyrics.json")
        print("\n  Run without --dry-run to continue the full pipeline.")
        return

    # ── Step 3: Treatment ────────────────────────────────────────────────────
    if start_idx <= STEPS.index("treatment"):
        print("\n[Step 3/6] Generating shot list with Claude…")
        try:
            shots = run_treatment()
        except Exception as e:
            print(f"ERROR in treatment: {e}")
            sys.exit(1)

        checkpoint(
            "Approve Shot List",
            f"  {len(shots)} shots printed above.\n"
            "  You can edit shots/treatment.json before continuing.\n"
            "  Approving will start generating keyframe images (costs ~$0.01–0.05 total).",
        )

    # ── Step 4: Generate frames ──────────────────────────────────────────────
    if start_idx <= STEPS.index("frames"):
        print("[Step 4/6] Generating keyframes with Flux Schnell…")
        try:
            frames_manifest = run_generate_frames()
        except Exception as e:
            print(f"ERROR in generate_frames: {e}")
            sys.exit(1)

        frames_dir = Path(SHOTS_DIR) / "frames"
        _open_folder(frames_dir)

        ok = frames_manifest["successful"]
        fail = frames_manifest["failed"]
        checkpoint(
            "Approve Visual Style",
            f"  Frames saved to: {frames_dir.absolute()}\n"
            f"  Successful: {ok}   Failed: {len(fail)}\n"
            + (f"  Failed shots: {fail}\n" if fail else "")
            + "  Open the folder and review the images.\n"
            "  Approving will generate 5-second video clips (costs ~$1–3 total).",
        )

    # ── Step 5: Generate clips ───────────────────────────────────────────────
    if start_idx <= STEPS.index("clips"):
        print("[Step 5/6] Generating video clips with Kling v1.6…")
        try:
            clips_manifest = run_generate_clips()
        except Exception as e:
            print(f"ERROR in generate_clips: {e}")
            sys.exit(1)

        if clips_manifest["failed"]:
            _prompt_regen_clips(clips_manifest)

        clips_dir = Path(SHOTS_DIR) / "clips"
        ok = clips_manifest["successful"]
        fail = clips_manifest["failed"]
        checkpoint(
            "Approve Clips & Assemble",
            f"  Clips saved to: {clips_dir.absolute()}\n"
            f"  Successful: {ok}   Failed: {len(fail)}\n"
            + (f"  Failed shots: {fail}\n" if fail else "")
            + "  Review any clips you like before approving.\n"
            "  Approving will assemble the final music video (no extra API cost).",
        )

    # ── Step 6: Assemble ─────────────────────────────────────────────────────
    if start_idx <= STEPS.index("assemble"):
        print("[Step 6/6] Assembling final video with moviepy…")
        try:
            output = run_assemble(song_path)
        except Exception as e:
            print(f"ERROR in assemble: {e}")
            sys.exit(1)

        print("\n" + "=" * width)
        print("  PIPELINE COMPLETE")
        print("=" * width)
        print(f"  Final video: {Path(output).absolute()}")
        print("=" * width)


if __name__ == "__main__":
    main()
