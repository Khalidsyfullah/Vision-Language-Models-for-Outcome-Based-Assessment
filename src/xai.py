"""Explainability: score-conditioned attention + deletion faithfulness.

Memory note: attention attribution needs the full LxHxTxT attention tensors, which
is O(layers * T^2). At T=4096 with 28 layers that is several GB, so XAI runs
with a reduced image side (study5.xai_max_image_side) and refuses sequences
longer than study5.max_seq_for_rollout instead of OOM-ing.

The model must be loaded with attn_implementation="eager"
(config: model_loading.attn_implementation) or attentions come back as None.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Config
from .inference import parse_score
from .io_utils import save_explanation_image, save_json
from .models import move_inputs_to_model, set_padding_side
from .prompting import apply_chat_template_safe


# --------------------------------------------------------------------------
# Attention rollout
# --------------------------------------------------------------------------

def _attention_rollout(attentions, target_index: int,
                       last_k_layers: int | None = None):
    """Roll attention from one output query back through the layers.

    Computing one target row is O(layers * sequence^2); multiplying complete
    rollout matrices would be O(layers * sequence^3) and is unnecessary.
    """
    import torch

    attentions = [a for a in attentions if a is not None]
    if not attentions:
        raise RuntimeError("No attention tensors — load the model with "
                           "attn_implementation='eager'.")
    if last_k_layers:
        attentions = attentions[-int(last_k_layers):]
    a = [att.mean(dim=1).float() for att in attentions]      # avg heads
    a = [m + torch.eye(m.size(-1), device=m.device).unsqueeze(0) for m in a]
    a = [m / m.sum(dim=-1, keepdim=True).clamp(min=1e-12) for m in a]
    seq_len = a[0].shape[-1]
    if not 0 <= int(target_index) < seq_len:
        raise IndexError(f"target attention index {target_index} outside "
                         f"sequence of length {seq_len}")
    relevance = torch.zeros(a[0].shape[0], seq_len, device=a[0].device,
                            dtype=a[0].dtype)
    relevance[:, int(target_index)] = 1.0
    for m in reversed(a):
        relevance = torch.bmm(relevance.unsqueeze(1), m).squeeze(1)
    return relevance


def _score_image_attention(attentions, target_index: int,
                           image_positions: list[int],
                           last_k_layers: int | None = 8,
                           baseline_queries: int = 4):
    """Direct score-query attention with positional-sink cancellation.

    Recursive causal rollout systematically accumulates relevance at the
    earliest image token: every later patch can route through that prefix
    position, producing the identical top-left hotspot seen in every supplied
    figure. Here each selected layer's score-query distribution over image
    keys is normalised, then the mean distribution of nearby pre-score queries
    is subtracted. Static first-token/border sinks cancel while attention that
    specifically increases when the numeric score is emitted is retained.
    """
    import torch

    layers = [a for a in attentions if a is not None]
    if not layers:
        raise RuntimeError("No attention tensors — load the model with "
                           "attn_implementation='eager'.")
    if last_k_layers:
        layers = layers[-int(last_k_layers):]
    if not image_positions:
        raise ValueError("image_positions cannot be empty")

    per_layer = []
    for attention in layers:
        matrix = attention.mean(dim=1).float()  # batch x query x key
        if int(target_index) >= matrix.shape[-2]:
            raise IndexError(
                f"target attention index {target_index} outside query length "
                f"{matrix.shape[-2]}")
        score = matrix[:, int(target_index), image_positions]
        score = score / score.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        n_base = max(0, int(baseline_queries))
        start = max(0, int(target_index) - n_base)
        if n_base and start < int(target_index):
            baseline = matrix[:, start:int(target_index), image_positions]
            baseline = baseline / baseline.sum(
                dim=-1, keepdim=True).clamp(min=1e-12)
            score = torch.relu(score - baseline.mean(dim=1))
        per_layer.append(score)

    relevance = torch.stack(per_layer).mean(dim=0)
    # If every positive difference cancels numerically, fall back to direct
    # score attention rather than emitting an all-zero explanation.
    if float(relevance.max()) <= 1e-12:
        direct = []
        for attention in layers:
            row = attention.mean(dim=1).float()[
                :, int(target_index), image_positions]
            direct.append(row / row.sum(dim=-1, keepdim=True).clamp(min=1e-12))
        relevance = torch.stack(direct).mean(dim=0)
    return relevance


def _img_token_ids(processor) -> list[int]:
    # Patch/context tokens only. Boundary and row-separator tokens must not be
    # treated as pixels when reconstructing the image grid.
    candidates = ["<image>", "<|image_pad|>", "[IMG]", "<|image|>",
                  "<IMG_CONTEXT>"]
    tok = getattr(processor, "tokenizer", None) or processor
    unk_id = getattr(tok, "unk_token_id", None)
    ids = set()
    for c in candidates:
        try:
            tid = tok.convert_tokens_to_ids(c)
            if isinstance(tid, int) and tid > 0 and tid != unk_id:
                ids.add(tid)
        except Exception:
            pass
    for attr in ("image_token_id", "image_token_index", "image_pad_token_id"):
        v = getattr(processor, attr, None)
        if isinstance(v, int) and v > 0:
            ids.add(v)
    return list(ids)


def _score_token_offset(tokenizer, token_ids, answer: str,
                        allow_bare: bool = False) -> tuple[int, float]:
    """Locate the generated token that completes the parsed numeric score."""
    final, _, _ = parse_score(answer, allow_bare=allow_bare)
    if final is None:
        raise RuntimeError(f"could not parse score from generated output: {answer!r}")
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    for end in range(1, len(ids) + 1):
        prefix = tokenizer.decode(ids[:end], skip_special_tokens=True)
        value, _, _ = parse_score(prefix, allow_bare=allow_bare)
        if value is not None and abs(value - final) < 1e-12:
            return end - 1, final
    raise RuntimeError("parsed score could not be aligned to generated tokens")


def _factor_grid(n_tokens: int, image) -> tuple[int, int]:
    """Exact factorization whose aspect ratio best matches the input image."""
    if n_tokens < 1:
        raise ValueError("image token count must be positive")
    target = image.height / max(1, image.width)
    candidates = []
    for gh in range(1, int(np.sqrt(n_tokens)) + 1):
        if n_tokens % gh == 0:
            gw = n_tokens // gh
            candidates.extend([(gh, gw), (gw, gh)])
    return min(candidates, key=lambda hw: abs(hw[0] / hw[1] - target))


def _image_grid(processor, inputs, n_tokens: int, image) -> tuple[int, int]:
    """Resolve the processor's image-token grid without square padding."""
    grid = inputs.get("image_grid_thw")
    if grid is not None:
        values = grid[0].detach().cpu().tolist()
        if len(values) == 3:
            _, gh, gw = (int(v) for v in values)
            merge = int(getattr(getattr(processor, "image_processor", None),
                                "merge_size", 1) or 1)
            gh, gw = max(1, gh // merge), max(1, gw // merge)
            if gh * gw == n_tokens:
                return gh, gw
    # Some processors do not expose a grid. An exact factorization preserves
    # every patch token and its overall aspect ratio; unlike the old ceil/square
    # path it neither pads nor drops attention values.
    return _factor_grid(n_tokens, image)


def _donut_cross_attention(cfg: Config, processor, model, image, example):
    """Donut explanation: decoder→encoder CROSS-attention.

    An encoder-decoder has no image tokens in its input sequence, so attention
    rollout does not apply. The natural relevance signal is how much the
    decoder attends to each encoder patch while emitting the score.
    """
    import torch
    import torch.nn.functional as F

    from .prompting import build_donut_prompt

    tok = processor.tokenizer
    task_token = str(cfg.get_in("donut.task_start_token", "<s_grade>"))
    prompt = task_token + build_donut_prompt(
        example, int(cfg.get_in("donut.max_prompt_chars", 1200)))
    pixel_values = move_inputs_to_model(
        {"pixel_values": processor(
            image, return_tensors="pt").pixel_values}, model)["pixel_values"]
    dec = tok(prompt, return_tensors="pt", add_special_tokens=False,
              truncation=True,
              max_length=int(cfg.get_in("donut.max_text_length", 512))
              ).input_ids.to(model.device)

    with torch.inference_mode():
        gen = model.generate(
            pixel_values=pixel_values, decoder_input_ids=dec,
            max_new_tokens=int(cfg.get_in("donut.max_new_tokens", 16)),
            do_sample=False)
        new_ids = gen[0, dec.shape[1]:]
        answer = tok.decode(new_ids, skip_special_tokens=True)
        offset, pred = _score_token_offset(tok, new_ids, answer,
                                           allow_bare=True)
        # Decoder position t predicts token t+1. Use the query that actually
        # generated the final score token, not the last prompt token by default
        # and not a state that has already consumed the score.
        target_query = max(0, dec.shape[1] + offset - 1)
        out = model(pixel_values=pixel_values, decoder_input_ids=gen,
                    decoder_attention_mask=torch.ones_like(gen),
                    output_attentions=True, return_dict=True)

    cross = getattr(out, "cross_attentions", None)
    if not cross:
        raise RuntimeError("Donut returned no cross-attentions")
    # Average over layers/heads at the decoder state that generated the score.
    att = torch.stack(
        [c.mean(dim=1)[:, target_query, :] for c in cross]).mean(0)[0]
    att = att.float().cpu().numpy()
    att = (att - att.min()) / (np.ptp(att) + 1e-12)

    # Preserve every Swin encoder position in an exact aspect-matched grid.
    gh, gw = _factor_grid(len(att), image)
    heat = F.interpolate(
        torch.tensor(att.reshape(gh, gw))[None, None],
        size=(image.height, image.width),
        mode="bilinear", align_corners=False)[0, 0].numpy()
    return heat, pred, answer


def _image_attention(cfg: Config, processor, model, image, user_text: str):
    """Return (heatmap HxW in [0,1], predicted score, raw answer)."""
    import torch
    import torch.nn.functional as F

    s5 = cfg.study5
    max_seq = int(s5.get("max_seq_for_rollout", 3072))
    method = str(s5.get("attribution_method",
                        "contrastive_score_attention"))
    last_k = s5.get("attention_last_k_layers",
                    s5.get("rollout_last_k_layers", 8))
    baseline_queries = int(s5.get("attention_baseline_queries", 4))

    prompt = apply_chat_template_safe(processor, user_text=user_text,
                                      add_generation_prompt=True)
    inp = move_inputs_to_model(
        processor(text=[prompt], images=[image], return_tensors="pt",
                  padding=True, truncation=False), model)
    prompt_len = inp["input_ids"].shape[1]

    with torch.inference_mode():
        gen_ids = model.generate(**inp, max_new_tokens=64,
                                 do_sample=False, use_cache=True)
        new_ids = gen_ids[0, prompt_len:]
        answer = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        offset, pred = _score_token_offset(processor.tokenizer, new_ids, answer)
        target_query = max(0, prompt_len + offset - 1)
        full_len = gen_ids.shape[1]
        if full_len > max_seq:
            raise RuntimeError(
                f"sequence too long for attention attribution "
                f"({full_len} > {max_seq}); "
                f"lower study5.xai_max_image_side or raise "
                f"study5.max_seq_for_rollout")

        forward_inp = dict(inp)
        forward_inp["input_ids"] = gen_ids
        forward_inp["attention_mask"] = torch.ones_like(gen_ids)
        if "token_type_ids" in forward_inp:
            tt = forward_inp["token_type_ids"]
            extra = tt[:, -1:].expand(-1, full_len - prompt_len)
            forward_inp["token_type_ids"] = torch.cat([tt, extra], dim=1)
        # Let the model recompute full-sequence positions.
        forward_inp.pop("position_ids", None)
        forward_inp.pop("cache_position", None)
        outputs = model(**forward_inp, output_attentions=True,
                        use_cache=False, return_dict=True)

    seq = gen_ids[0, :prompt_len].tolist()
    img_ids = set(_img_token_ids(processor))
    img_pos = [i for i, t in enumerate(seq) if t in img_ids]
    if len(img_pos) < 4:
        raise RuntimeError(
            f"only {len(img_pos)} image tokens located — the processor's "
            f"image token id was not recognised for this model")
    if method == "rollout":
        # Retained for reproducibility of legacy results; decoder-only causal
        # rollout is not the default because it creates prefix attention sinks.
        relevance = _attention_rollout(
            outputs.attentions, target_query, last_k)[0, img_pos]
    elif method in ("score_attention", "contrastive_score_attention"):
        relevance = _score_image_attention(
            outputs.attentions, target_query, img_pos, last_k,
            baseline_queries=(baseline_queries if method.startswith(
                "contrastive") else 0))[0]
    else:
        raise ValueError(
            f"unsupported study5.attribution_method={method!r}; use "
            "contrastive_score_attention, score_attention, or rollout")
    del outputs
    attn = relevance.float().cpu().numpy()
    # Percentile clipping prevents one remaining outlier patch from flattening
    # every other spatial difference in the visualisation.
    low, high = np.percentile(attn, [1.0, 99.0])
    if high <= low:
        low, high = float(attn.min()), float(attn.max())
    attn = np.clip((attn - low) / (high - low + 1e-12), 0, 1)
    gh, gw = _image_grid(processor, inp, len(attn), image)
    heat = F.interpolate(
        torch.tensor(attn.reshape(gh, gw))[None, None],
        size=(image.height, image.width),
        mode="bilinear", align_corners=False)[0, 0].numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return heat, pred, answer


def _explain_one(cfg, processor, model, item, donut: bool):
    """Family dispatch for a single item."""
    if donut:
        return _donut_cross_attention(cfg, processor, model,
                                      item["image"], item["example"])
    return _image_attention(cfg, processor, model, item["image"],
                            item["user_text"])


def explain_samples(cfg: Config, processor, model, dataset,
                    out_dir: Path, n_samples: int,
                    donut: bool = False) -> list[dict]:
    """Score-attention heatmap figures (answer / heatmap / overlay)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm

    print(f"\n=== XAI heatmaps ({n_samples} samples) ===")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_padding_side(processor, "left")
    n = min(int(n_samples), len(dataset))
    idxs = np.random.default_rng(cfg.seed).choice(len(dataset), n,
                                                  replace=False)

    meta, failures = [], 0
    for idx in tqdm(idxs.tolist(), desc="xai"):
        item = dataset[idx]
        ex, image = item["example"], item["image"]
        try:
            heat, pred, answer = _explain_one(cfg, processor, model,
                                              item, donut)
            fig, ax = plt.subplots(1, 3, figsize=(15, 7))
            ax[0].imshow(image); ax[0].set_title("Student answer")
            method_key = ("donut_score_cross_attention" if donut else
                          str(cfg.study5.get(
                              "attribution_method",
                              "contrastive_score_attention")))
            method_label = method_key.replace("_", " ").title()
            ax[1].imshow(heat, cmap="jet"); ax[1].set_title(method_label)
            ax[2].imshow(image); ax[2].imshow(heat, cmap="jet", alpha=0.45)
            ax[2].set_title(f"Overlay (pred {pred} / "
                            f"true {float(ex['gained_marks']):g})")
            for a in ax:
                a.axis("off")
            fig.suptitle(f"{ex['criterion_name'][:70]} — {ex['example_id']}",
                         fontsize=11)
            fig.tight_layout()
            save_explanation_image(fig, out_dir / str(ex["example_id"]))
            meta.append({"example_id": ex["example_id"],
                         "answer_id": ex["answer_id"],
                         "model": cfg.study5.xai_model,
                         "regime": cfg.study5.xai_regime,
                         "criterion_name": ex["criterion_name"],
                         "pred_score": pred,
                         "true_score": float(ex["gained_marks"]),
                         "max_score": float(ex.get("resolved_max") or 0),
                         "model_output": answer.strip(),
                         "attribution_method": method_key,
                         "figure": f"{ex['example_id']}.png"})
        except Exception as e:  # keep going on individual failures
            failures += 1
            print(f"[XAI] skip {ex['example_id']}: {e}")
    print(f"  produced {len(meta)} heatmaps ({failures} failed)")
    save_json(meta, out_dir / "xai_index.json")
    return meta


# --------------------------------------------------------------------------
# Faithfulness (deletion metric, Study 5)
# --------------------------------------------------------------------------

def _mask_top_regions(image, heat: np.ndarray, fraction: float,
                      invert: bool = False):
    """Grey out the top `fraction` most-attended pixels (or the complement)."""
    from PIL import Image as PILImage

    arr = np.asarray(image.convert("RGB")).copy()
    thresh = np.quantile(heat, 1 - fraction)
    mask = heat >= thresh
    if invert:
        mask = ~mask
    arr[mask] = 127
    return PILImage.fromarray(arr)


def _score_once(cfg, processor, model, image, user_text: str,
                example: dict | None = None,
                donut: bool = False) -> float | None:
    import torch

    if donut:
        from .prompting import build_donut_prompt

        tok = processor.tokenizer
        task_token = str(cfg.get_in("donut.task_start_token", "<s_grade>"))
        prompt = task_token + build_donut_prompt(
            example, int(cfg.get_in("donut.max_prompt_chars", 1200)))
        pv = move_inputs_to_model(
            {"pixel_values": processor(
                image, return_tensors="pt").pixel_values}, model)["pixel_values"]
        dec = tok(prompt, return_tensors="pt", add_special_tokens=False,
                  truncation=True,
                  max_length=int(cfg.get_in("donut.max_text_length", 512))
                  ).input_ids.to(model.device)
        with torch.inference_mode():
            gen = model.generate(pv, decoder_input_ids=dec,
                                 max_new_tokens=int(cfg.get_in(
                                     "donut.max_new_tokens", 16)))
        txt = tok.batch_decode(gen[:, dec.shape[1]:],
                               skip_special_tokens=True)[0]
        return parse_score(txt, allow_bare=True)[0]

    prompt = apply_chat_template_safe(processor, user_text=user_text,
                                      add_generation_prompt=True)
    inp = move_inputs_to_model(
        processor(text=[prompt], images=[image], return_tensors="pt",
                  padding=True, truncation=False), model)
    if inp["input_ids"].shape[1] > int(cfg.inference.max_length):
        raise RuntimeError(
            f"XAI scoring prompt has {inp['input_ids'].shape[1]} tokens, above "
            f"inference.max_length={cfg.inference.max_length}; reduce the image "
            f"size instead of truncating multimodal placeholder tokens")
    with torch.inference_mode():
        gen = model.generate(**inp,
                             max_new_tokens=cfg.inference.max_new_tokens,
                             do_sample=False, use_cache=True)
    txt = processor.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0]
    return parse_score(txt)[0]


def faithfulness_eval(cfg: Config, processor, model, dataset,
                      out_dir: Path, donut: bool = False) -> list[dict]:
    """Deletion faithfulness: if the explanation is faithful, masking the
    most-attended regions should change the score MORE than masking random
    regions of the same size.

    Per sample and mask fraction:
      score_full, score_del_top, score_del_random,
      comprehensiveness = |score_full - score_del_top|
      random_effect     = |score_full - score_del_random|
      faithful          = comprehensiveness > random_effect
    """
    from tqdm.auto import tqdm

    fcfg = cfg.study5.faithfulness
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_padding_side(processor, "left")
    rng = np.random.default_rng(cfg.seed)
    n = min(int(fcfg.n_samples), len(dataset))
    idxs = rng.choice(len(dataset), n, replace=False)

    rows = []
    for idx in tqdm(idxs.tolist(), desc="faithfulness"):
        item = dataset[idx]
        ex, image = item["example"], item["image"]
        try:
            heat, _, _ = _explain_one(cfg, processor, model, item, donut)
            score_full = _score_once(cfg, processor, model, image,
                                     item["user_text"], item["example"], donut)
            if score_full is None:
                continue
            rand_heat = rng.random(heat.shape)
            for frac in fcfg.mask_fractions:
                frac = float(frac)
                s_top = _score_once(cfg, processor, model,
                                    _mask_top_regions(image, heat, frac),
                                    item["user_text"], item["example"], donut)
                s_rand = _score_once(cfg, processor, model,
                                     _mask_top_regions(image, rand_heat, frac),
                                     item["user_text"], item["example"], donut)
                comp = abs(score_full - s_top) if s_top is not None else None
                rand_eff = (abs(score_full - s_rand)
                            if s_rand is not None else None)
                rows.append({
                    "example_id": ex["example_id"],
                    "criterion_name": ex["criterion_name"],
                    "mask_fraction": frac,
                    "score_full": score_full,
                    "score_del_top": s_top,
                    "score_del_random": s_rand,
                    "comprehensiveness": comp,
                    "random_effect": rand_eff,
                    "faithful": (comp is not None and rand_eff is not None
                                 and comp > rand_eff),
                })
        except Exception as e:
            print(f"[faithfulness] skip {ex['example_id']}: {e}")

    (out_dir / "faithfulness_raw.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows
