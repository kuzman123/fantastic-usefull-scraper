# Fantastic-usefull-scraper
A multi-stage pipeline for extracting structured, AI-enhanced Markdown documents from YouTube tutorials. Each output .md file contains timestamped transcription, visual frame descriptions, and chapter-organized content — optimized for knowledge base ingestion, retrieval-augmented systems, and structured content pipelines
## Overview

The pipeline runs across two systems:

- **Windows** — Stage 1: video download, audio extraction, frame extraction (no GPU required)
- **Linux / Cloud GPU** — Stages 2–4: transcription, deduplication, vision description, final assembly

YouTube Download — Windows vs. Linux

- **Stage 1 (video download) is intentionally designed to run on Windows. Cloud GPU providers assign IP addresses that YouTube frequently flags as automated traffic, resulting in bot-detection blocks (Sign in to confirm you're not a bot). Residential IPs on Windows machines are not subject to this restriction under normal usage.
You can to Stage 1 on Linux Cloud GPU also. I've checked multiple times and it works, except once when I was blocked. All from the same YouTube account.

Both stages share the same `output/` folder structure with relative paths, making it fully portable across operating systems without any configuration changes.

---

## Pipeline Stages

| Stage | Script | Runs on | Description |
|-------|--------|---------|-------------|
| 1 | `stage1_download.py` | Windows or Linux | Download video, extract audio (16kHz mono WAV), extract frames (scene-change or interval) |
| 2 | `stage2_transcribe.py` | Linux / GPU | Transcribe audio with faster-whisper, group by sentence |
| 2b | `stage2b_dedup.py` | Linux / GPU | CLIP-based frame deduplication (enabled by default, optional) |
| 3 | `stage3_vision.py` | Linux / GPU | Describe each frame using Qwen2.5-VL vision model |
| 4 | `stage4_assemble.py` | Linux / GPU | Assemble final `.md` with timestamps, IMAGE descriptions, and chapter sections |

---

## Requirements

### Python
- Python 3.10 (required — tested stable)

### CUDA & PyTorch (tested stable combinations)

| Environment | CUDA Driver | PyTorch | Torchvision | xformers |
|-------------|------------|---------|-------------|----------|
| Windows (Stage 1 only) | Any | Not required | Not required | Not required |
| Linux CUDA 12.6 | 12.6+ | 2.7.1+cu126 | 0.22.1+cu126 | 0.0.30 (cu126) |
| Linux CUDA 12.8 | 12.8+ | 2.7.1+cu128 | 0.22.1+cu128 | Not compatible with py3.10 |

> **Important:** xformers is only compatible with the cu126 build on Python 3.10. On cu128 + Python 3.10, xformers has no compatible wheel — set `USE_XFORMERS = False` in `config.py` and the pipeline falls back to SDPA automatically.

> **Do NOT install bitsandbytes on Volta-architecture GPUs (V100)** — causes segfault on import.

### System dependency
```bash
# Linux
apt update && apt install -y ffmpeg

# Windows
# Download ffmpeg from https://ffmpeg.org and add to PATH
```

---

## Installation

### Windows (Stage 1 only)

```cmd
git clone fantastic-usefull-scraper
cd C:\fantastic-usefull-scraper
python -m venv venv
venv\Scripts\activate
pip install yt-dlp pillow
```

### Linux / Cloud GPU (Stages 2–4)

```bash
cd /workspace
python -m venv venv
source venv/bin/activate

# PyTorch — install FIRST, before everything else
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

# Core dependencies
pip install faster-whisper "transformers>=4.49.0,<5.0" accelerate \
    "qwen-vl-utils[decord]" opencv-python pillow huggingface_hub

# xformers (cu126 only — skip if using cu128)
pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu126

pip install yt-dlp

# ffmpeg
apt update && apt install -y ffmpeg
```

### Installing ffmpeg on Windows
1. Download
Go to https://www.gyan.dev/ffmpeg/builds/ and download ffmpeg-release-essentials.zip (stable release).
2. Extract
Extract the zip to C:\ffmpeg. Avoid paths with spaces or special characters (Desktop, Downloads, user folders). The result should look like:

```cmd
C:\ffmpeg\
    bin\
        ffmpeg.exe
        ffprobe.exe
        ffplay.exe
```

4. Add to System PATH

Press Win+S, search for "Edit the system environment variables", press Enter
Click "Environment Variables..."
In the System variables section (bottom panel — not User variables), select Path → click Edit
Click New → enter C:\ffmpeg\bin
Click OK on all open dialogs

Use System variables, not User variables. This ensures ffmpeg is accessible to all processes, scheduled tasks, and scripts regardless of which user account runs them.

4. Verify
Open a new Command Prompt window (important — existing windows won't see the updated PATH) and run:
cmdffmpeg -version
Expected output starts with ffmpeg version 7.x.x .... If you see 'ffmpeg' is not recognized, restart the Command Prompt and try again.


### Download models (Linux, once per environment)

```bash
cd /workspace/fantastic-usefull-scraper
python download_models.py
```

Downloads:
- `faster-whisper large-v3` (~3 GB) → `models/whisper/`
- `Qwen2.5-VL-7B-Instruct` (~16 GB) → `models/qwen_vl/`
- `CLIP ViT-L/14` (~900 MB) → `models/clip/`

---

## Usage

### Combined Windows + Linux workflow (recommended)

**On Windows:**
```cmd
cd C:\fantastic-usefull-scraper
venv\Scripts\activate

# Add YouTube URLs to urls.txt (one per line, # for comments)
python stage1_download.py

tar -czf output.tar.gz output
```

Upload `output.tar.gz` to your Linux/cloud environment.

**On Linux:**
```bash
source /workspace/venv/bin/activate
cd /workspace/fantastic-usefull-scraper

tar -xzf output.tar.gz
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_all.py
```

Final `.md` files are in `output/md/`.

---

### Linux-only workflow (all stages)

```bash
source /workspace/venv/bin/activate
cd /workspace/fantastic-usefull-scraper

# Add URLs to urls.txt
python stage1_download.py
python run_all.py
```

---

### Windows-only workflow (Stage 1 only, no GPU)

Stage 1 runs entirely on CPU via ffmpeg. Useful for pre-processing video locally before transferring to a GPU machine for Stages 2–4.

```cmd
python stage1_download.py
```

Output folders (`audio/`, `frames/`, `logs/`) are portable — copy them to any Linux machine and continue from Stage 2.

---

## Output Structure

```
output/
├── audio/       # 16kHz mono WAV files (Whisper input)
├── frames/      # JPEG frames (timestamp-named: frame_00012340ms.jpg)
├── logs/        # Per-video JSON metadata (schema v2)
└── md/          # Final .md files  ← primary output
```

All paths inside `logs/*.json` are stored as relative paths — the `output/` folder can be moved between Windows and Linux without modification.

### Final .md structure

Each output `.md` contains:
- Video metadata header (URL, uploader, duration, date)
- Chapter or time-based sections
- Timestamped transcription segments
- `[IMAGE: ...]` blocks with AI-generated visual descriptions at scene-change moments

This format is well-suited for knowledge base ingestion, retrieval-augmented generation (RAG) pipelines, semantic search indexing, and structured documentation workflows.

---

## Tweakable Parameters

All parameters are in `config.py`.

### Frame Extraction

| Parameter | Default | Effect |
|-----------|---------|--------|
| `FRAME_EXTRACTION_MODE` | `"scene_change"` | `"scene_change"` extracts frames at detected visual changes. `"interval"` extracts at fixed time intervals. Scene-change produces fewer, more meaningful frames; interval guarantees uniform coverage. |
| `SCENE_CHANGE_THRESHOLD` | `0.3` | Sensitivity of scene-change detection (0.0–1.0). Lower = more frames captured, higher = fewer. For fast-paced screen content, `0.2`–`0.25` is recommended. |
| `MIN_FRAME_GAP_SECONDS` | `3` | Minimum time between two extracted frames. Prevents burst extraction on rapid transitions. |
| `MAX_FRAMES_PER_VIDEO` | `300` | Hard cap on frames per video. Increase for longer videos with dense visual content. |
| `FRAME_MAX_DIMENSION` | `1536` | Maximum size of the longer edge (pixels). Larger = better OCR and detail in visual descriptions, higher VRAM usage. Recommended range: `1280`–`1920`. |
| `FRAME_MIN_DIMENSION` | `768` | Minimum size of the longer edge. Frames below this are upscaled using Lanczos. Prevents very small frames from degrading vision model accuracy. |
| `FRAME_JPEG_QUALITY` | `2` | ffmpeg JPEG quality (1 = best, 31 = worst). `1`–`3` is visually lossless. |

### Deduplication (Stage 2b)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ENABLE_DEDUP` | `True` | Enable CLIP-based frame deduplication. Disabled, all scene-change frames are passed to Stage 3. Set to `False` to skip this stage entirely and process all extracted frames. |
| `DEDUP_THRESHOLD` | `0.95` | Cosine similarity cutoff for marking a frame as duplicate (0.0–1.0). Higher = stricter, keeps more frames. Lower = more aggressive removal. `0.90`–`0.97` is the practical range. |
| `DEDUP_TIME_WINDOW_SEC` | `10` | Only compare frames within this time window. Prevents comparing unrelated scenes across the video. |

### Vision Model (Stage 3)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `VISION_MODEL` | `"Qwen/Qwen2.5-VL-7B-Instruct"` | Vision model used for frame description. `7B` fits in 24 GB VRAM. `32B` requires 2× 32 GB or quantization. `72B` requires 3× 32 GB or 80 GB. Larger = more accurate descriptions and better OCR of on-screen text. |
| `VISION_BATCH_SIZE` | `1` | Number of frames processed simultaneously by the vision model. Higher = faster Stage 3, higher VRAM usage. On RTX 3090 (24 GB) with 7B model at 1536px: `2`–`3` is stable. On 1920px: `1`–`2`. |
| `VISION_MAX_NEW_TOKENS` | `150` | Maximum tokens in each generated frame description. Higher = more detailed output, slower inference. `150` produces 1–2 sentences; `250` produces 2–4 sentences with parameter values. |
| `VISION_DTYPE` | `"bfloat16"` | Model compute dtype. `bfloat16` is optimal for Ampere/Ada (RTX 30xx, 40xx). `float16` for older architectures. |
| `USE_XFORMERS` | `True` | Enable xformers memory-efficient attention. Reduces VRAM usage ~15% and speeds inference. Requires compatible build — see Requirements table. Falls back to SDPA if unavailable. |
| `USE_MULTI_GPU` | `False` | Shard model across multiple GPUs using `device_map="balanced"`. Useful for 32B+ models that exceed single GPU capacity. Has overhead on small models — leave `False` for 7B. |
| `USE_QUANTIZATION` | `False` | 4-bit NF4 quantization via bitsandbytes. Reduces model VRAM ~50%. Not compatible with Volta (V100) architecture. |

### Transcription (Stage 2)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `WHISPER_MODEL` | `"large-v3"` | Whisper model size. `large-v3` is the highest quality available. Smaller models (`medium`, `small`) are faster but less accurate, particularly for technical vocabulary. |
| `WHISPER_COMPUTE_TYPE` | `"float16"` | Compute type for faster-whisper. `float16` is optimal for CUDA. `int8_float16` reduces VRAM at minor quality cost. |
| `USE_VAD_FILTER` | `True` | Enable Voice Activity Detection. Skips silence segments, speeds up transcription, and reduces hallucination on quiet sections. |
| `VAD_MIN_SILENCE_MS` | `200` | Minimum silence duration (ms) for VAD to trigger a segment split. Lower values catch shorter pauses, producing more granular segments. Increase to `500` for cleaner cuts with less fragmentation. |
| `WHISPER_SEGMENT_MODE` | `"sentence"` | How transcription segments are grouped. `"sentence"` groups by punctuation. `"chunk_30s"` groups into 30-second blocks. `"raw"` uses Whisper's native segmentation. |

### Output Assembly (Stage 4)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `GROUP_INTO_SECTIONS` | `True` | Group content into YouTube chapters if available, otherwise into time blocks. Produces structured, navigable output. |
| `FALLBACK_SECTION_MINUTES` | `5` | Time block size (minutes) used when YouTube chapters are absent. |
| `INCLUDE_METADATA_HEADER` | `True` | Prepend video metadata (URL, uploader, date, duration, description excerpt) to the output `.md`. |

---

## VRAM Reference (7B model)

| Frame size | Batch size | Estimated VRAM |
|------------|------------|----------------|
| 1280px | 3 | ~18–20 GB |
| 1536px | 2 | ~20–22 GB |
| 1536px | 3 | ~22–24 GB |
| 1920px | 1 | ~18–20 GB |
| 1920px | 2 | ~22–24 GB |

RTX 3090 (24 GB) is the reference GPU. Adjust `VISION_BATCH_SIZE` down if OOM errors appear.

---

## Environment Variable (recommended)

Set before running Stage 3 to reduce VRAM fragmentation:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Tested Environments

| OS | GPU | CUDA | PyTorch | Status |
|----|-----|------|---------|--------|
| Windows 10/11 | Any (Stage 1 only) | N/A | N/A | ✅ Stable |
| Ubuntu 22.04 | RTX 3090 | 12.8 | 2.7.1+cu128 | ✅ Stable |
| Ubuntu 22.04 | RTX 3090 | 12.6 | 2.7.1+cu126 | ✅ Stable |
| Ubuntu 22.04 | V100 32GB | 12.6 | 2.7.1+cu126 | ✅ Stable (no bitsandbytes) |
| Ubuntu 22.04 | 2× V100 32GB | 12.6 | 2.7.1+cu126 | ✅ Stable (USE_MULTI_GPU=True, 32B only) |
