# Music Video Agent

An AI-powered pipeline that turns a song file into a finished music video with human-in-the-loop approval at every expensive step.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY and FAL_KEY

# Test your setup with no API costs
python run.py --song path/to/song.mp3 --dry-run

# Run the full pipeline
python run.py --song path/to/song.mp3
```

## Pipeline Overview

```
song.mp3
   │
   ├─ [Step 1] analyze.py     → shots/analysis.json    (BPM, beats, sections, energy)
   ├─ [Step 2] transcribe.py  → shots/lyrics.json      (word-level timestamps)
   │
   │  ◆ CHECKPOINT: Review shot list ◆
   │
   ├─ [Step 3] treatment.py   → shots/treatment.json   (20-25 shots via Claude)
   │
   │  ◆ CHECKPOINT: Approve visual style ◆
   │
   ├─ [Step 4] generate_frames.py → shots/frames/      (keyframe PNGs via Flux Schnell)
   │
   │  ◆ CHECKPOINT: Approve clips & assemble ◆
   │
   ├─ [Step 5] generate_clips.py  → shots/clips/       (5s MP4s via Kling v1.6)
   │
   └─ [Step 6] assemble.py        → output/final.mp4   (moviepy cut + audio)
```

## Checkpoints

The pipeline pauses three times for human approval before spending money on generation:

| Checkpoint | What to review | Approving triggers |
|---|---|---|
| **Approve Shot List** | The 20-25 shots printed to terminal; edit `shots/treatment.json` if needed | Keyframe image generation (~$0.01–0.05) |
| **Approve Visual Style** | PNG images in `shots/frames/`; open the folder in your file manager | Video clip generation (~$1–3) |
| **Approve Clips & Assemble** | MP4 clips in `shots/clips/`; review in any video player | Final assembly (free — local only) |

At each checkpoint, type:
- `y` or Enter — approve and continue
- `q` — stop here (run with `--start-from` to resume later)

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (Steps 3+) | Anthropic API key for Claude |
| `FAL_KEY` | Yes (Steps 4–5) | fal.ai API key for Flux + Kling |

## CLI Reference

### `run.py` — Main orchestrator

```bash
python run.py --song SONG [--dry-run] [--start-from STEP]
```

| Flag | Description |
|---|---|
| `--song PATH` | Path to input audio file (MP3, WAV, FLAC, etc.) |
| `--dry-run` | Run Steps 1–2 only (analysis + transcription, zero API cost) |
| `--start-from STEP` | Resume from a step: `analyze`, `transcribe`, `treatment`, `frames`, `clips`, `assemble` |

### Individual modules

Each module can also be run standalone:

```bash
# Re-analyze audio
python analyze.py song.mp3

# Re-transcribe
python transcribe.py song.mp3 --model small

# Regenerate shot list
python treatment.py --shots 24

# Regenerate all frames (skip existing by default)
python generate_frames.py

# Regenerate specific frames by shot number
python generate_frames.py --regen 3 7 12

# Regenerate all clips
python generate_clips.py

# Regenerate specific clips
python generate_clips.py --regen 3 7 12

# Force-regenerate everything (ignore existing files)
python generate_frames.py --no-resume
python generate_clips.py --no-resume

# Re-assemble from existing clips
python assemble.py --song song.mp3 --crossfade 0.5
```

## Regenerating Individual Shots

If a few clips look weak after generation:

1. **Edit the prompt** in `shots/treatment.json` for the shots you want to change
2. **Regenerate frames** for those shots:
   ```bash
   python generate_frames.py --regen 5 11 18
   ```
3. **Regenerate clips** for those shots:
   ```bash
   python generate_clips.py --regen 5 11 18
   ```
4. **Re-assemble** (fast, no extra API cost):
   ```bash
   python assemble.py --song song.mp3
   ```

Or resume the full pipeline from the clips step:
```bash
python run.py --song song.mp3 --start-from clips
```

## Output Files

| Path | Description |
|---|---|
| `shots/analysis.json` | BPM, beats, sections, energy curve |
| `shots/lyrics.json` | Full transcription with word timestamps |
| `shots/treatment.json` | Shot list with prompts (edit this to customize shots) |
| `shots/frames/shot_NNN.png` | Keyframe image for each shot |
| `shots/clips/shot_NNN.mp4` | 5-second video clip for each shot |
| `shots/frames_manifest.json` | Frame generation status log |
| `shots/clips_manifest.json` | Clip generation status log |
| `output/final.mp4` | Final assembled music video |

## Models Used

| Step | Model | Service | Notes |
|---|---|---|---|
| Transcription | Whisper `base` | Local (free) | Downloads ~150 MB on first run |
| Shot list | `claude-sonnet-4-20250514` | Anthropic API | ~$0.003 per run |
| Keyframes | `fal-ai/flux/schnell` | fal.ai | ~$0.003 per image |
| Video clips | `fal-ai/kling-video/v1.6/standard/image-to-video` | fal.ai | ~$0.05–0.14 per clip |

## Troubleshooting

**`openai-whisper` import error**: Install with `pip install openai-whisper`. You may also need `pip install torch`.

**`librosa` section detection returns too few sections**: This is normal for very short tracks (<60s). The pipeline adapts to the number of detected sections.

**fal.ai clip generation times out**: Kling clips take 60–120s each. If a clip fails, it will retry up to 3 times with exponential backoff. Use `--regen` to retry specific shots.

**`moviepy` errors on assembly**: Ensure `ffmpeg` is installed on your system (`brew install ffmpeg` / `apt install ffmpeg`). The project pins `moviepy<2.0` for API stability.

**Resuming a stopped pipeline**: Use `--start-from STEP`. All intermediate files are preserved in `shots/`, so you only pay for steps that haven't completed yet.
