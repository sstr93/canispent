"""
Step 5 — Video clip generation.
For each approved keyframe, calls Kling v1.6 (standard, image-to-video) via
fal.ai to generate a 5-second clip. Saves MP4 files to shots/clips/.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import fal_client


def generate_clip(shot: dict, frame_path: Path, out_path: Path, retries: int = 3) -> bool:
    """Generate one video clip from a keyframe. Returns True on success."""
    # Combine prompt + camera motion for a richer motion directive
    motion_prompt = f"{shot['prompt']} Camera: {shot['camera_motion']}."

    # Upload frame to fal storage to get a reachable URL
    try:
        image_url = fal_client.upload_file(str(frame_path))
    except Exception as e:
        print(f"    Upload failed: {e}")
        return False

    for attempt in range(retries):
        try:
            result = fal_client.subscribe(
                "fal-ai/kling-video/v1.6/standard/image-to-video",
                arguments={
                    "prompt": motion_prompt,
                    "image_url": image_url,
                    "duration": "5",
                    "aspect_ratio": "16:9",
                },
            )
            video_url = result["video"]["url"]
            urllib.request.urlretrieve(video_url, str(out_path))
            return True

        except Exception as e:
            wait = 2**attempt
            print(f"    Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                print(f"    Retrying in {wait}s…")
                time.sleep(wait)

    return False


def generate_all_clips(output_dir: str = "shots", resume: bool = True) -> dict:
    treatment_path = Path(output_dir) / "treatment.json"
    frames_dir = Path(output_dir) / "frames"
    clips_dir = Path(output_dir) / "clips"

    if not treatment_path.exists():
        raise FileNotFoundError(f"{treatment_path} not found — run treatment.py first.")

    treatment = json.loads(treatment_path.read_text())
    shots = treatment["shots"]
    clips_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(shots)} clips with Kling v1.6 → {clips_dir}")
    print("(Each clip takes ~60–120s on fal.ai)\n")

    results: dict[int, dict] = {}

    for i, shot in enumerate(shots, 1):
        n = shot["shot_number"]
        frame_path = frames_dir / f"shot_{n:03d}.png"
        out_path = clips_dir / f"shot_{n:03d}.mp4"

        if not frame_path.exists():
            print(f"  [{i:>2}/{len(shots)}] Shot {n}: SKIP — frame not found ({frame_path.name})")
            results[n] = {"status": "no_frame", "path": None}
            continue

        if resume and out_path.exists():
            print(f"  [{i:>2}/{len(shots)}] Shot {n}: already exists, skipping")
            results[n] = {"status": "skipped", "path": str(out_path)}
            continue

        print(f"  [{i:>2}/{len(shots)}] Shot {n}: {shot['description'][:65]}…")
        ok = generate_clip(shot, frame_path, out_path)

        if ok:
            print(f"           → {out_path.name}")
            results[n] = {"status": "success", "path": str(out_path)}
        else:
            print(f"           → FAILED")
            results[n] = {"status": "failed", "path": None}

    success_count = sum(1 for r in results.values() if r["status"] in ("success", "skipped"))
    failed = [n for n, r in results.items() if r["status"] == "failed"]
    no_frame = [n for n, r in results.items() if r["status"] == "no_frame"]

    manifest = {
        "clips_dir": str(clips_dir),
        "results": {str(k): v for k, v in results.items()},
        "successful": success_count,
        "failed": failed,
        "no_frame": no_frame,
    }
    (Path(output_dir) / "clips_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nClip generation complete: {success_count}/{len(shots)} succeeded")
    if failed:
        print(f"  Failed shot numbers: {failed}")
    if no_frame:
        print(f"  Missing frames:      {no_frame}")

    return manifest


def regenerate_clips(shot_numbers: list[int], output_dir: str = "shots") -> None:
    """Force-regenerate specific clips by shot number."""
    treatment = json.loads((Path(output_dir) / "treatment.json").read_text())
    shot_map = {s["shot_number"]: s for s in treatment["shots"]}
    frames_dir = Path(output_dir) / "frames"
    clips_dir = Path(output_dir) / "clips"

    for n in shot_numbers:
        if n not in shot_map:
            print(f"Shot {n} not found in treatment.")
            continue
        frame_path = frames_dir / f"shot_{n:03d}.png"
        out_path = clips_dir / f"shot_{n:03d}.mp4"
        out_path.unlink(missing_ok=True)
        print(f"Regenerating shot {n}…")
        ok = generate_clip(shot_map[n], frame_path, out_path)
        print("  Success" if ok else "  Failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video clips with Kling v1.6")
    parser.add_argument("--output-dir", default="shots")
    parser.add_argument("--no-resume", action="store_true", help="Regenerate all clips, including existing ones")
    parser.add_argument("--regen", nargs="+", type=int, metavar="N", help="Regenerate specific shot numbers only")
    args = parser.parse_args()

    if args.regen:
        regenerate_clips(args.regen, args.output_dir)
    else:
        generate_all_clips(args.output_dir, resume=not args.no_resume)
