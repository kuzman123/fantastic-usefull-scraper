"""stage4_assemble.py v2.1 - skip UNCLEAR"""
import json, re, sys
from pathlib import Path
from datetime import datetime
import config


def safe_filename(name, max_len=120):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name.strip('._')[:max_len]


def format_timestamp(seconds):
    total = int(seconds)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    if h > 0:
        return config.TIMESTAMP_FORMAT.replace("{mm:02d}:{ss:02d}", f"{h}:{m:02d}:{s:02d}")
    return config.TIMESTAMP_FORMAT.format(mm=m, ss=s)


def merge_events(transcript, frames):
    events = []
    for seg in transcript:
        events.append({'type': 'transcript', 'timestamp': seg['start'], 'content': seg['text'].strip()})
    for f in frames:
        if not f.get('dedup_kept', True):
            continue
        desc = f.get('description')
        if not desc or desc == "[[UNCLEAR]]":
            continue
        events.append({'type': 'image', 'timestamp': f['timestamp_sec'], 'content': desc})
    events.sort(key=lambda e: e['timestamp'])
    return events


def apply_chapters(events, chapters):
    if not chapters:
        return [{'title': None, 'start': 0, 'events': events}]
    sections = []
    for ch in chapters:
        sections.append({'title': ch.get('title', 'Untitled'), 'start': ch['start_time'],
                         'end': ch.get('end_time', float('inf')), 'events': []})
    for ev in events:
        for sec in sections:
            if sec['start'] <= ev['timestamp'] < sec['end']:
                sec['events'].append(ev)
                break
    return sections


def apply_time_sections(events, minutes):
    if not events:
        return []
    chunk_sec = minutes * 60
    sections = []
    current_start = 0
    current_events = []
    for ev in events:
        if ev['timestamp'] >= current_start + chunk_sec:
            if current_events:
                sections.append({'title': f"{current_start//60}-{current_start//60 + minutes} min",
                                 'start': current_start, 'events': current_events})
            current_start += chunk_sec
            while ev['timestamp'] >= current_start + chunk_sec:
                current_start += chunk_sec
            current_events = []
        current_events.append(ev)
    if current_events:
        sections.append({'title': f"{current_start//60}-{current_start//60 + minutes} min",
                         'start': current_start, 'events': current_events})
    return sections


def render_md(meta):
    lines = [f"# {meta['title'].replace('_', ' ')}", ""]
    if config.INCLUDE_METADATA_HEADER:
        lines += ["## Video Info", ""]
        lines.append(f"- **URL:** {meta['url']}")
        lines.append(f"- **Video ID:** {meta['video_id']}")
        if meta.get('uploader'):
            lines.append(f"- **Uploader:** {meta['uploader']}")
        if meta.get('upload_date'):
            d = meta['upload_date']
            lines.append(f"- **Upload date:** {d[:4]}-{d[4:6]}-{d[6:8]}")
        dur = meta.get('duration', 0)
        lines.append(f"- **Duration:** {dur//60}:{dur%60:02d}")
        lines.append(f"- **Scraped:** {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("")
        if meta.get('description'):
            desc = meta['description'][:300].strip()
            if desc:
                lines += ["**Description:**", f"> {desc}", ""]
        lines += ["---", ""]

    events = merge_events(meta.get('transcript', []), meta.get('frames', []))
    if config.GROUP_INTO_SECTIONS:
        sections = apply_chapters(events, meta['chapters']) if meta.get('chapters') \
                   else apply_time_sections(events, config.FALLBACK_SECTION_MINUTES)
    else:
        sections = [{'title': None, 'start': 0, 'events': events}]

    lines += ["## Content", ""]
    for sec in sections:
        if sec['title']:
            lines += [f"### {sec['title']}", ""]
        for ev in sec['events']:
            ts = format_timestamp(ev['timestamp'])
            if ev['type'] == 'image':
                lines.append(f"{ts} [IMAGE: {ev['content']}]")
            else:
                lines.append(f"{ts} {ev['content']}")
            lines.append("")
    return '\n'.join(lines)


def main():
    config.MD_DIR.mkdir(parents=True, exist_ok=True)
    log_files = list(config.LOGS_DIR.glob('*.json'))
    if not log_files:
        print(f"No metadata files")
        sys.exit(1)
    for log_file in log_files:
        meta = json.loads(log_file.read_text(encoding='utf-8'))
        if 'transcript' not in meta:
            print(f"SKIP (no transcript): {meta['title']}")
            continue
        md_content = render_md(meta)
        md_path = config.MD_DIR / f"{safe_filename(meta['title'])}.md"
        md_path.write_text(md_content, encoding='utf-8')
        print(f"OK: {md_path.name} ({md_path.stat().st_size // 1024} KB)")


if __name__ == '__main__':
    main()
