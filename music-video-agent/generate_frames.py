"""
Step 4 — Keyframe image generation.
Calls Flux Schnell via fal.ai for each shot in the treatment.
Saves PNG images to shots/frames/shot_NNN.png.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import fal_client


def generate_frame(shot: dict, out_path: Path, retries: int = 3) -> bool:
    """Generate one keyframe image. Returns True on success."""
    for attempt in range(retries):
        try:
            result = fal_client.subscribe(
                "fal-ai/flux/schnell",
                arguments={
                    "prompt": shot["prompt"],
                    "image_size": "landscape_16_9",
                    "num_inference_steps": 4,
                    "num_images": 1,
                    "enable_safety_checker": True,
                },
            )
            image_url = result["images"][0]["url"]
            urllib.request.urlretrieve(image_url, str(out_path))
            return True

        except Exception as e:
            wait = 2**attempt
            print(f"    Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                print(f"    Retrying in {wait}s…")
                time.sleep(wait)

    return False


def generate_all_frames(output_dir: str = "shots", resume: bool = True) -> dict:
    treatment_path = Path(output_dir) / "treatment.json"
    if not treatment_path.exists():
        raise FileNotFoundError(f"{treatment_path} not found — run treatment.py first.")

    treatment = json.loads(treatment_path.read_text())
    shots = treatment["shots"]
    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(shots)} keyframes with Flux Schnell → {frames_dir}")
    print("(Each image takes ~5–15s)\n")

    results: dict[int, dict] = {}

    for i, shot in enumerate(shots, 1):
        n = shot["shot_number"]
        out_path = frames_dir / f"shot_{n:03d}.png"

        if resume and out_path.exists():
            print(f"  [{i:>2}/{len(shots)}] Shot {n}: already exists, skipping")
            results[n] = {"status": "skipped", "path": str(out_path)}
            continue

        print(f"  [{i:>2}/{len(shots)}] Shot {n}: {shot['description'][:65]}…")
        ok = generate_frame(shot, out_path)

        if ok:
            print(f"           → {out_path.name}")
            results[n] = {"status": "success", "path": str(out_path)}
        else:
            print(f"           → FAILED")
            results[n] = {"status": "failed", "path": None}

    success_count = sum(1 for r in results.values() if r["status"] in ("success", "skipped"))
    failed = [n for n, r in results.items() if r["status"] == "failed"]

    manifest = {
        "frames_dir": str(frames_dir),
        "results": {str(k): v for k, v in results.items()},
        "successful": success_count,
        "failed": failed,
    }
    manifest_path = Path(output_dir) / "frames_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nFrame generation complete: {success_count}/{len(shots)} succeeded")
    if failed:
        print(f"  Failed shot numbers: {failed}")
        print("  Re-run generate_frames.py --regen to retry failed shots only.")

    return manifest


def regenerate_frames(shot_numbers: list[int], output_dir: str = "shots") -> None:
    """Regenerate specific shots by number."""
    treatment = json.loads((Path(output_dir) / "treatment.json").read_text())
    shot_map = {s["shot_number"]: s for s in treatment["shots"]}
    frames_dir = Path(output_dir) / "frames"

    for n in shot_numbers:
        if n not in shot_map:
            print(f"Shot {n} not found in treatment.")
            continue
        out_path = frames_dir / f"shot_{n:03d}.png"
        out_path.unlink(missing_ok=True)
        print(f"Regenerating shot {n}…")
        ok = generate_frame(shot_map[n], out_path)
        print("  Success" if ok else "  Failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate keyframes with Flux Schnell")
    parser.add_argument("--output-dir", default="shots")
    parser.add_argument("--no-resume", action="store_true", help="Regenerate all frames, including existing ones")
    parser.add_argument("--regen", nargs="+", type=int, metavar="N", help="Regenerate specific shot numbers only")
    args = parser.parse_args()

    if args.regen:
        regenerate_frames(args.regen, args.output_dir)
    else:
        generate_all_frames(args.output_dir, resume=not args.no_resume)
