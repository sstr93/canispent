"""
Step 2 — Lyrics transcription.
Uses OpenAI Whisper (local, base model) to transcribe lyrics with
word-level timestamps. Saves results to shots/lyrics.json.
"""

import argparse
import json
from pathlib import Path


def transcribe_audio(
    song_path: str,
    output_dir: str = "shots",
    model_size: str = "base",
) -> dict:
    try:
        import whisper
    except ImportError:
        raise ImportError("openai-whisper is not installed. Run: pip install openai-whisper")

    print(f"Loading Whisper model ({model_size}) — first run downloads ~150 MB…")
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        print(f"  WARNING: Could not load Whisper model: {e}")
        return _empty_lyrics(song_path, output_dir)

    print(f"Transcribing: {song_path}")
    try:
        result = model.transcribe(str(song_path), word_timestamps=True, verbose=False)
    except Exception as e:
        print(f"  WARNING: Transcription failed: {e}")
        return _empty_lyrics(song_path, output_dir)

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append(
                {
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                }
            )

    segments = [
        {
            "text": seg["text"].strip(),
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
        }
        for seg in result.get("segments", [])
    ]

    lyrics = {
        "song_path": str(song_path),
        "language": result.get("language", "unknown"),
        "full_text": result.get("text", "").strip(),
        "segments": segments,
        "words": words,
    }

    out = Path(output_dir) / "lyrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lyrics, indent=2))

    print(f"  Language: {lyrics['language']}")
    print(f"  Words:    {len(words)}")
    print(f"  Segments: {len(segments)}")
    if lyrics["full_text"]:
        preview = lyrics["full_text"][:200]
        print(f"\n  Preview: {preview}{'…' if len(lyrics['full_text']) > 200 else ''}")
    else:
        print("  (No lyrics detected — instrumental or quiet track)")
    print(f"Saved: {out}")
    return lyrics


def _empty_lyrics(song_path: str, output_dir: str) -> dict:
    """Write a stub lyrics.json when Whisper is unavailable."""
    lyrics = {
        "song_path": str(song_path),
        "language": "unknown",
        "full_text": "",
        "segments": [],
        "words": [],
        "_note": "Transcription skipped — Whisper model unavailable. Edit this file to add lyrics manually.",
    }
    out = Path(output_dir) / "lyrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lyrics, indent=2))
    print("  Saved empty lyrics stub. You can edit shots/lyrics.json to add lyrics manually.")
    print(f"Saved: {out}")
    return lyrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("song", help="Path to the song file")
    parser.add_argument("--output-dir", default="shots")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (larger = slower but more accurate)",
    )
    args = parser.parse_args()
    transcribe_audio(args.song, args.output_dir, args.model)
