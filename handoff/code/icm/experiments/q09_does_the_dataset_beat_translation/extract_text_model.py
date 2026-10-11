"""Q09 helper (CPU): turn a vision+text checkpoint (Gemma 3, Gemma3ForConditionalGeneration)
into a text-only one (Gemma3ForCausalLM), so the student is trained and evaluated
exactly like Qwen and Mistral: plain AutoModelForCausalLM, LoRA on the usual
projection names, vLLM's text model.

Tensors are COPIED, not changed: every key under the language model's prefix is
renamed to the text model's prefix; the vision tower and the projector are left
out. The script refuses to write unless every text-model key is accounted for,
and checks that no language-model tensor is dropped.

    python -m icm.experiments.q09_does_the_dataset_beat_translation.extract_text_model \
        --src <gemma-3-12b-pt dir> --dst <gemma-3-12b-pt-text dir>
"""
import argparse
import json
import shutil
from pathlib import Path

# checkpoint prefix of the language model -> prefix Gemma3ForCausalLM expects
PREFIXES = (("language_model.model.", "model."), ("model.language_model.", "model."),
            ("language_model.lm_head.", "lm_head."))
DROP = ("vision_tower.", "multi_modal_projector.", "model.vision_tower.", "model.multi_modal_projector.")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "generation_config.json")


def rename(key):
    for old, new in PREFIXES:
        if key.startswith(old):
            return new + key[len(old):]
    if key.startswith(DROP):
        return None
    raise KeyError(f"unexpected key {key!r}: neither language model nor vision")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig, Gemma3ForCausalLM

    text_cfg = AutoConfig.from_pretrained(src).text_config
    text_cfg.architectures = ["Gemma3ForCausalLM"]
    text_cfg.torch_dtype = "bfloat16"
    with torch.device("meta"):
        expected = set(Gemma3ForCausalLM(text_cfg).state_dict())

    index = json.loads((src / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    weight_map, kept, dropped = {}, set(), 0
    for i, shard in enumerate(shards):
        out = {}
        with safe_open(src / shard, framework="pt") as f:
            for k in f.keys():
                nk = rename(k)
                if nk is None:
                    dropped += 1
                    continue
                if nk in kept:
                    raise SystemExit(f"duplicate key after renaming: {nk}")
                out[nk] = f.get_tensor(k)
                kept.add(nk)
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(out, dst / name, metadata={"format": "pt"})
        weight_map.update({k: name for k in out})
        print(f"{shard}: {len(out)} text tensors -> {name}", flush=True)

    tied = {"lm_head.weight"} if text_cfg.tie_word_embeddings else set()
    missing, extra = expected - kept - tied, kept - expected
    if missing or extra:
        for p in dst.glob("model-*.safetensors"):
            p.unlink()
        raise SystemExit(f"key mismatch, nothing written. missing {sorted(missing)[:5]} ({len(missing)}), "
                         f"extra {sorted(extra)[:5]} ({len(extra)})")
    total = sum((dst / n).stat().st_size for n in set(weight_map.values()))
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))
    text_cfg.save_pretrained(dst)
    for f in TOKENIZER_FILES:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    print(f"text model: {len(kept)} tensors kept, {dropped} vision/projector tensors dropped; "
          f"all {len(expected)} expected keys present (lm_head tied: {bool(tied)}). wrote {dst}")


if __name__ == "__main__":
    main()
