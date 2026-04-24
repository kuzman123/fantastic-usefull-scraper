"""
download_models.py v2 - Downloads Whisper + Qwen2.5-VL + (optional) CLIP

Run once before first use.
CLIP is downloaded only if ENABLE_DEDUP = True (default).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config


def download_whisper():
    print("="*70)
    print(" faster-whisper large-v3 (~3GB)")
    print("="*70)
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("ERROR: faster-whisper is not installed!")
        return False

    whisper_dir = config.MODELS_DIR / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading in: {whisper_dir}")
    try:
        model = WhisperModel(
            config.WHISPER_MODEL, device="cpu", compute_type="int8",
            download_root=str(whisper_dir),
        )
        del model
        print("OK.\n")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_qwen():
    print("="*70)
    print(f" {config.VISION_MODEL} (~16GB)")
    print("="*70)
    print("20-30 min depending on internet speed.\n")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed!")
        return False

    qwen_dir = config.MODELS_DIR / "qwen_vl"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading in: {qwen_dir}")
    try:
        snapshot_download(
            repo_id=config.VISION_MODEL,
            local_dir=str(qwen_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.onnx"],
        )
        print("OK.\n")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_clip():
    print("="*70)
    print(f" {config.DEDUP_MODEL} (~350MB)")
    print("="*70)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed!")
        return False

    clip_dir = config.MODELS_DIR / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading in: {clip_dir}")
    try:
        snapshot_download(
            repo_id=config.DEDUP_MODEL,
            local_dir=str(clip_dir),
            local_dir_use_symlinks=False,
        )
        print("OK.\n")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nModels in: {config.MODELS_DIR}\n")

    success = True
    success &= download_whisper()
    success &= download_qwen()
    success &= download_clip()

    if not success:
        print("Some models failed.")
        sys.exit(1)
    print("\nOK.")


if __name__ == '__main__':
    main()
