"""stage2_transcribe.py v2.1 - relative paths"""
import json, sys
from pathlib import Path
from faster_whisper import WhisperModel
import config


def load_whisper():
    print(f"Loading faster-whisper: {config.WHISPER_MODEL}")
    print(f"Device: {config.WHISPER_DEVICE}, compute_type: {config.WHISPER_COMPUTE_TYPE}")
    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        download_root=str(config.WHISPER_MODEL_DIR),
    )
    print("Model loaded.\n")
    return model


def group_segments_by_sentence(segments):
    grouped = []
    buf_text = []
    buf_start = None
    last_end = 0
    for seg in segments:
        text = seg['text'].strip()
        if not text:
            continue
        if buf_start is None:
            buf_start = seg['start']
        buf_text.append(text)
        last_end = seg['end']
        if text and text[-1] in '.!?':
            grouped.append({'start': buf_start, 'end': last_end, 'text': ' '.join(buf_text)})
            buf_text = []
            buf_start = None
    if buf_text:
        grouped.append({'start': buf_start, 'end': last_end, 'text': ' '.join(buf_text)})
    return grouped


def group_segments_by_chunks(segments, chunk_size=30.0):
    grouped = []
    current_start = 0
    current_text = []
    last_end = 0
    for seg in segments:
        if seg['start'] >= current_start + chunk_size and current_text:
            grouped.append({'start': current_start, 'end': seg['start'], 'text': ' '.join(current_text)})
            current_start = seg['start']
            current_text = []
        current_text.append(seg['text'].strip())
        last_end = seg['end']
    if current_text:
        grouped.append({'start': current_start, 'end': last_end, 'text': ' '.join(current_text)})
    return grouped


def transcribe_audio(model, audio_path):
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=config.WHISPER_LANGUAGE,
        beam_size=config.WHISPER_BEAM_SIZE,
        vad_filter=config.USE_VAD_FILTER,
        vad_parameters=dict(min_silence_duration_ms=config.VAD_MIN_SILENCE_MS) if config.USE_VAD_FILTER else None,
        condition_on_previous_text=True,
    )
    raw_segments = []
    for seg in segments_iter:
        raw_segments.append({'start': seg.start, 'end': seg.end, 'text': seg.text})
    if config.WHISPER_SEGMENT_MODE == "sentence":
        return group_segments_by_sentence(raw_segments)
    elif config.WHISPER_SEGMENT_MODE == "chunk_30s":
        return group_segments_by_chunks(raw_segments, 30.0)
    return raw_segments


def main():
    model = load_whisper()
    log_files = list(config.LOGS_DIR.glob('*.json'))
    if not log_files:
        print(f"No metadata files in {config.LOGS_DIR}")
        sys.exit(1)
    print(f"Founded {len(log_files)} video(a)\n")

    for log_file in log_files:
        meta = json.loads(log_file.read_text(encoding='utf-8'))
        if 'transcript' in meta and meta['transcript']:
            print(f"SKIP: {meta['title']} (already transcribed)")
            continue
        audio_path = config.resolve_path(meta['audio_path'])
        if not audio_path.exists():
            print(f"ERROR: Audio is not exist: {audio_path}")
            continue
        print(f"Transcribing: {meta['title']}")
        try:
            segments = transcribe_audio(model, audio_path)
            meta['transcript'] = segments
            log_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f"  OK: {len(segments)} segment(a)\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")


if __name__ == '__main__':
    main()
