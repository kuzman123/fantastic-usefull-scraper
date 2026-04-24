"""stage3_vision.py v2.1 - relative paths, fixed description cleanup"""
import json, sys, re
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import config


def load_vision_model():
    print(f"Loading Vision: {config.VISION_MODEL}")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(config.VISION_DTYPE, torch.bfloat16)

    attn_impl = "sdpa"
    if getattr(config, 'USE_XFORMERS', False):
        try:
            import xformers
            print(f"  xformers: available (through SDPA)")
        except ImportError:
            print(f"  xformers: not installed, SDPA")

    model_path = str(config.QWEN_MODEL_DIR) if config.QWEN_MODEL_DIR.exists() else config.VISION_MODEL
    print(f"  Model path: {model_path}")
    print(f"  Dtype: {config.VISION_DTYPE}")

    load_kwargs = {"torch_dtype": dtype, "attn_implementation": attn_impl}

    if getattr(config, 'USE_MULTI_GPU', False) and torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        max_mem = getattr(config, 'MAX_MEMORY_PER_GPU', "30GB")
        load_kwargs["device_map"] = "balanced"
        load_kwargs["max_memory"] = {i: max_mem for i in range(n_gpus)}
        print(f"  Multi-GPU: {n_gpus}x {max_mem}")
    else:
        load_kwargs["device_map"] = config.VISION_DEVICE
        print(f"  Single GPU")

    if getattr(config, 'USE_QUANTIZATION', False):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        load_kwargs.pop("torch_dtype", None)
        print(f"  Quantization: 4-bit")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)

    if torch.cuda.is_available():
        vram = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 1024**3
        cap = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1024**3
        print(f"  VRAM: {vram:.1f} / {cap:.1f} GB\n")
    return model, processor


def find_transcript_context(ts_sec, transcript, window=10.0):
    if not transcript:
        return ""
    relevant = []
    for seg in transcript:
        if abs(seg['start'] - ts_sec) < window or (seg['start'] <= ts_sec <= seg['end']):
            relevant.append(seg['text'].strip())
    if not relevant and transcript:
        closest = min(transcript, key=lambda s: abs(s['start'] - ts_sec))
        relevant.append(closest['text'].strip())
    return ' '.join(relevant)[:500]


def _clean_description(d):
    d = re.sub(r'\baddCriterion[^\n.]*\.?', '', d, flags=re.IGNORECASE)
    d = re.sub(r'\bExample\s*:?\s*\n?', '', d)
    d = re.sub(r'\n\s*\n', '\n', d).strip()
    if "Description unavailable" in d or "[[UNCLEAR]]" in d:
        return "[[UNCLEAR]]"
    if not d or len(d) < 20:
        return "[[UNCLEAR]]"
    return d


def describe_images_batch(model, processor, image_contexts):
    messages_batch = []
    for img_path, ctx in image_contexts:
        if ctx and config.USE_TRANSCRIPT_CONTEXT_FOR_VISION:
            prompt = config.VISION_PROMPT_WITH_CONTEXT.format(context=ctx)
        else:
            prompt = config.VISION_PROMPT
        messages_batch.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": prompt},
            ],
        }])
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
    all_images = []
    for m in messages_batch:
        img_inputs, _ = process_vision_info(m)
        all_images.extend(img_inputs)
    inputs = processor(text=texts, images=all_images, padding=True, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=config.VISION_MAX_NEW_TOKENS, do_sample=False)
    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    descriptions = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [d.strip() for d in descriptions]


def describe_single(model, processor, img_path, ctx=""):
    return describe_images_batch(model, processor, [(img_path, ctx)])[0]


def main():
    model, processor = load_vision_model()
    log_files = list(config.LOGS_DIR.glob('*.json'))
    if not log_files:
        print(f"No metadata files")
        sys.exit(1)

    for log_file in log_files:
        meta = json.loads(log_file.read_text(encoding='utf-8'))
        if 'transcript' not in meta:
            print(f"SKIP: {meta['title']} (no transcript)")
            continue
        frames = meta.get('frames', [])
        if not frames:
            continue
        todo_frames = [f for f in frames
                       if f.get('dedup_kept', True) and f.get('description') is None]
        if not todo_frames:
            print(f"SKIP: {meta['title']} (all described or deduped)")
            continue
        print(f"\nDescribing: {meta['title']}")
        print(f"  {len(todo_frames)}/{len(frames)} frame(s)")

        image_contexts = []
        for f in todo_frames:
            ctx = find_transcript_context(f['timestamp_sec'], meta['transcript'])
            image_contexts.append((config.resolve_path(f['path']), ctx))

        batch_size = config.VISION_BATCH_SIZE
        idx_cursor = 0
        pbar = tqdm(range(0, len(image_contexts), batch_size), desc="  Vision")
        for i in pbar:
            batch = image_contexts[i:i + batch_size]
            try:
                descs = describe_images_batch(model, processor, batch)
            except torch.cuda.OutOfMemoryError:
                print(f"\n  OOM, fallback single...")
                torch.cuda.empty_cache()
                descs = []
                for img_path, ctx in batch:
                    try:
                        descs.append(describe_single(model, processor, img_path, ctx))
                    except Exception:
                        descs.append("[[UNCLEAR]]")
            except Exception as e:
                print(f"\n  Batch error: {e}")
                descs = []
                for img_path, ctx in batch:
                    try:
                        descs.append(describe_single(model, processor, img_path, ctx))
                    except Exception:
                        descs.append("[[UNCLEAR]]")
            for d in descs:
                todo_frames[idx_cursor]['description'] = _clean_description(d)
                idx_cursor += 1

        log_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"  OK: {idx_cursor} of description")
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
