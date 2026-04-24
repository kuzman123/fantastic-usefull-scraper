"""stage2b_dedup.py v2.1 - relative paths"""
import json, sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config


def main():
    if not config.ENABLE_DEDUP:
        print("Dedup disabled. Skipping.")
        return
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image

    model_path = str(config.CLIP_MODEL_DIR) if config.CLIP_MODEL_DIR.exists() else config.DEDUP_MODEL
    print(f"Loading CLIP: {model_path}")
    model = CLIPModel.from_pretrained(model_path).to(config.DEDUP_DEVICE)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_path)
    print("CLIP loaded.\n")

    log_files = list(config.LOGS_DIR.glob('*.json'))
    if not log_files:
        print(f"No metadata files")
        sys.exit(1)

    for log_file in log_files:
        meta = json.loads(log_file.read_text(encoding='utf-8'))
        frames = meta.get('frames', [])
        if not frames:
            continue
        if all(f.get('embedding') is not None for f in frames):
            if all(f.get('dedup_kept') is not None for f in frames):
                print(f"SKIP: {meta['title']} (deduped)")
                continue

        print(f"\nDedup for: {meta['title']}")
        print(f"  {len(frames)} frame(s)")

        embeddings = _compute_embeddings(frames, model, processor)
        for f, emb in zip(frames, embeddings):
            f['embedding'] = emb.tolist() if emb is not None else None

        kept = _mark_dedup(frames, config.DEDUP_THRESHOLD, config.DEDUP_TIME_WINDOW_SEC)
        print(f"  Withheld: {kept}, deduped: {len(frames) - kept}")
        log_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')


def _compute_embeddings(frames, model, processor):
    import torch
    from PIL import Image
    embeddings = [None] * len(frames)
    batch_size = config.DEDUP_BATCH_SIZE

    for start in tqdm(range(0, len(frames), batch_size), desc="  CLIP"):
        end = min(start + batch_size, len(frames))
        batch = frames[start:end]
        images = []
        indices = []
        for i, f in enumerate(batch):
            try:
                img_path = config.resolve_path(f['path'])
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                indices.append(start + i)
            except Exception:
                pass
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt").to(config.DEDUP_DEVICE)
        with torch.inference_mode():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_cpu = feats.cpu().numpy()
        for idx, feat in zip(indices, feats_cpu):
            embeddings[idx] = feat
    return embeddings


def _mark_dedup(frames, threshold, time_window_sec):
    kept_count = 0
    last_kept = []
    for f in frames:
        if f.get('embedding') is None:
            f['dedup_kept'] = True
            kept_count += 1
            continue
        t = f['timestamp_sec']
        emb = np.asarray(f['embedding'])
        last_kept = [(lt, le) for lt, le in last_kept if t - lt <= time_window_sec]
        is_dup = False
        for lt, le in last_kept:
            sim = float(np.dot(emb, le))
            if sim > threshold:
                is_dup = True
                break
        if is_dup:
            f['dedup_kept'] = False
        else:
            f['dedup_kept'] = True
            last_kept.append((t, emb))
            kept_count += 1
    return kept_count


if __name__ == '__main__':
    main()
