"""stage1_download.py v2.1 - relative paths"""
import os, re, sys, json, subprocess
from pathlib import Path
from PIL import Image
import config


def safe_filename(name, max_len=120):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name.strip('._')[:max_len]


def read_urls(urls_file):
    if not urls_file.exists():
        print(f"ERROR: No file {urls_file}")
        sys.exit(1)
    urls = []
    for line in urls_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        urls.append(line)
    return urls


def get_video_info(url):
    cmd = ['yt-dlp', '--dump-json', '--no-playlist']
    if config.YT_DLP_USER_AGENT:
        cmd += ['--user-agent', config.YT_DLP_USER_AGENT]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {result.stderr}")
    return json.loads(result.stdout)


def get_video_resolution(video_path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height',
           '-of', 'csv=p=0:s=x', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return (0, 0)
    try:
        w, h = result.stdout.strip().split('x')
        return (int(w), int(h))
    except Exception:
        return (0, 0)


def download_video(url, output_dir, video_id):
    output_template = str(output_dir / f"{video_id}.%(ext)s")
    cmd = ['yt-dlp', '-f', config.VIDEO_FORMAT,
           '--merge-output-format', 'mp4', '--no-playlist',
           '-o', output_template]
    if config.YT_DLP_USER_AGENT:
        cmd += ['--user-agent', config.YT_DLP_USER_AGENT]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr}")
    mp4_path = output_dir / f"{video_id}.mp4"
    if mp4_path.exists():
        return mp4_path
    for ext in ['.mp4', '.webm', '.mkv']:
        c = output_dir / f"{video_id}{ext}"
        if c.exists():
            return c
    raise FileNotFoundError("Video file not found after download")


def extract_audio(video_path, audio_path):
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-y', '-i', str(video_path),
           '-vn', '-ac', '1', '-ar', '16000',
           '-acodec', 'pcm_s16le', str(audio_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio failed: {result.stderr}")


def _build_scale_filter(src_w, src_h):
    max_d = config.FRAME_MAX_DIMENSION
    min_d = config.FRAME_MIN_DIMENSION
    flags = config.FRAME_SCALE_FLAGS
    longer = max(src_w, src_h)
    if src_w == 0 or src_h == 0:
        return f"scale='if(gt(iw,ih),{max_d},-2)':'if(gt(iw,ih),-2,{max_d})':flags={flags}"
    if longer > max_d:
        target = max_d
    elif longer < min_d:
        target = min_d
    else:
        return f"scale={src_w}:{src_h}:flags={flags}"
    if src_w >= src_h:
        return f"scale={target}:-2:flags={flags}"
    return f"scale=-2:{target}:flags={flags}"


def _parse_scene_timestamps(video_path):
    cmd = ['ffmpeg', '-i', str(video_path),
           '-filter:v', f"select='gt(scene,{config.SCENE_CHANGE_THRESHOLD})',showinfo",
           '-f', 'null', '-']
    result = subprocess.run(cmd, capture_output=True, text=True)
    timestamps = []
    for line in result.stderr.splitlines():
        m = re.search(r'pts_time:([\d.]+)', line)
        if m:
            timestamps.append(float(m.group(1)))
    filtered = []
    last_t = -999
    for t in timestamps:
        if t - last_t >= config.MIN_FRAME_GAP_SECONDS:
            filtered.append(t)
            last_t = t
        if len(filtered) >= config.MAX_FRAMES_PER_VIDEO:
            break
    if not filtered or filtered[0] > 2:
        filtered.insert(0, 0.5)
    return filtered


def _extract_single_frame(video_path, timestamp_sec, frame_path, scale_filter):
    cmd = ['ffmpeg', '-y', '-ss', str(timestamp_sec), '-i', str(video_path),
           '-frames:v', '1', '-vf', scale_filter,
           '-q:v', str(config.FRAME_JPEG_QUALITY), str(frame_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and frame_path.exists() and frame_path.stat().st_size > 0


def _validate_frame(frame_path):
    try:
        with Image.open(frame_path) as img:
            img.verify()
        with Image.open(frame_path) as img:
            w, h = img.size
            if w < 10 or h < 10:
                return None
            return {'width': w, 'height': h}
    except Exception:
        return None


def _frame_filename(timestamp_sec):
    ms = int(round(timestamp_sec * 1000))
    return f"frame_{ms:08d}ms.jpg"


def _relative_to_root(abs_path):
    """Convert absolute path to relative (string) relative to ROOT_DIR."""
    try:
        return str(Path(abs_path).relative_to(config.ROOT_DIR)).replace('\\', '/')
    except ValueError:
        return str(abs_path).replace('\\', '/')


def extract_frames(video_path, frames_dir, src_w, src_h):
    frames_dir.mkdir(parents=True, exist_ok=True)
    scale_filter = _build_scale_filter(src_w, src_h)

    if config.FRAME_EXTRACTION_MODE == "scene_change":
        timestamps = _parse_scene_timestamps(video_path)
        method = "scene_change"
    else:
        dur_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                   '-of', 'csv=p=0', str(video_path)]
        dur_result = subprocess.run(dur_cmd, capture_output=True, text=True)
        try:
            duration = float(dur_result.stdout.strip())
        except Exception:
            duration = 0
        timestamps = []
        t = 0.5
        while t < duration and len(timestamps) < config.MAX_FRAMES_PER_VIDEO:
            timestamps.append(t)
            t += config.FRAME_INTERVAL_SECONDS
        method = "interval"

    frame_records = []
    skipped_extract = 0
    skipped_decode = 0

    for t in timestamps:
        fname = _frame_filename(t)
        fpath = frames_dir / fname

        if not _extract_single_frame(video_path, t, fpath, scale_filter):
            skipped_extract += 1
            continue

        if config.VALIDATE_FRAMES_AFTER_EXTRACT:
            v = _validate_frame(fpath)
            if v is None:
                skipped_decode += 1
                try:
                    fpath.unlink()
                except Exception:
                    pass
                continue
            actual_w, actual_h = v['width'], v['height']
        else:
            actual_w, actual_h = 0, 0

        record = {
            'timestamp_ms': int(round(t * 1000)),
            'timestamp_sec': round(t, 3),
            'path': _relative_to_root(fpath),  # RELATIVE
            'filename': fname,
            'width': actual_w,
            'height': actual_h,
            'extraction_method': method,
            'scene_score': None,
            'decode_valid': True,
            'embedding': None,
            'dedup_kept': True,
        }
        frame_records.append(record)

    if skipped_extract:
        print(f"      WARNING: {skipped_extract} frames not exported")
    if skipped_decode:
        print(f"      WARNING: {skipped_decode} frames not passed decode validation")

    return frame_records


def process_video(url):
    print(f"\n{'='*70}")
    print(f"Processing: {url}")
    print(f"{'='*70}")

    info = get_video_info(url)
    video_id = info['id']
    title = safe_filename(info['title'])
    duration = info.get('duration', 0)
    print(f"Title: {title}")
    print(f"Duration: {duration//60}:{duration%60:02d}")

    md_path = config.MD_DIR / f"{title}.md"
    if config.SKIP_EXISTING and md_path.exists():
        print(f"SKIP: {md_path.name} already exists")
        return

    video_workdir = config.VIDEOS_DIR / video_id
    video_workdir.mkdir(parents=True, exist_ok=True)
    audio_path = config.AUDIO_DIR / f"{video_id}.wav"
    frames_workdir = config.FRAMES_DIR / video_id

    print("[1/3] Downloading...")
    video_path = download_video(url, video_workdir, video_id)
    src_w, src_h = get_video_resolution(video_path)
    print(f"      OK: {video_path.name} ({src_w}x{src_h})")

    print("[2/3] Extracting audio...")
    extract_audio(video_path, audio_path)
    print(f"      OK: {audio_path.name}")

    print("[3/3] Extracting frames...")
    frames = extract_frames(video_path, frames_workdir, src_w, src_h)
    print(f"      OK: {len(frames)} valid frame(s)")

    meta = {
        'schema_version': 2,
        'url': url,
        'video_id': video_id,
        'title': title,
        'duration': duration,
        'upload_date': info.get('upload_date'),
        'uploader': info.get('uploader'),
        'description': info.get('description', '')[:500],
        'chapters': info.get('chapters', []),
        'source_resolution': {'width': src_w, 'height': src_h},
        'frame_extraction': {
            'mode': config.FRAME_EXTRACTION_MODE,
            'max_dim': config.FRAME_MAX_DIMENSION,
            'min_dim': config.FRAME_MIN_DIMENSION,
            'scale_flags': config.FRAME_SCALE_FLAGS,
        },
        'audio_path': _relative_to_root(audio_path),  # RELATIVE
        'frames': frames,
    }
    meta_path = config.LOGS_DIR / f"{video_id}.json"
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    if config.DELETE_VIDEO_AFTER_PROCESSING:
        if video_path.exists():
            video_path.unlink()
        for f in video_workdir.iterdir():
            f.unlink()
        video_workdir.rmdir()


def main():
    for d in [config.OUTPUT_DIR, config.VIDEOS_DIR, config.AUDIO_DIR,
              config.FRAMES_DIR, config.MD_DIR, config.LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    urls = read_urls(config.URLS_FILE)
    print(f"Loaded {len(urls)} URLs\n")
    failed = []
    for url in urls:
        try:
            process_video(url)
        except Exception as e:
            print(f"\nERROR for {url}: {e}")
            failed.append((url, str(e)))
    print(f"\n{'='*70}\nSTAGE 1 FINISHED\n{'='*70}")
    print(f"Successfully: {len(urls) - len(failed)}/{len(urls)}")
    if failed:
        print("Unsuccessful:")
        for url, err in failed:
            print(f"  - {url}: {err[:100]}")


if __name__ == '__main__':
    main()
