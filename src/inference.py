"""Batched generation + robust score parsing (all regimes)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .config import Config
from .io_utils import write_records
from .models import is_donut, move_inputs_to_model, set_padding_side
from .prompting import apply_chat_template_safe, build_donut_prompt
from .scoring import clamp

# Robust to: {"score": 3}, "score": 3.5, ```json fenced```, trailing prose.
_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)
_JUST_RE = re.compile(r'"justification"\s*:\s*"([^"]*)"', re.IGNORECASE)
_JSON_BLOCK_RE = re.compile(r'\{[^{}]*"score"[^{}]*\}', re.DOTALL | re.IGNORECASE)
# bare fallback: "2.5/4", "Score: 2.5", "2.5 marks"
_BARE_RE = re.compile(r'(-?\d+(?:\.\d+)?)\s*(?:/\s*\d+(?:\.\d+)?|marks?\b)',
                      re.IGNORECASE)


_LEADING_NUM_RE = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*$')


def _finite_score(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) else None


def parse_score(text: str, allow_bare: bool = False
                ) -> tuple[float | None, str, bool]:
    """Return (score, justification, parsed_as_clean_json).

    allow_bare: accept a response that is nothing but a number ("2.5"). Donut
    is trained to emit exactly that, whereas for the chat VLMs a bare number
    would too easily match stray digits in prose.
    """
    if not text:
        return None, "", False
    if allow_bare:
        m = _LEADING_NUM_RE.match(text)
        if m:
            score = _finite_score(m.group(1))
            if score is not None:
                return score, "", True
    s = text.strip()
    for prefix in ("```json", "```JSON", "```"):
        if s.startswith(prefix):
            s = s[len(prefix):].lstrip("\n")
    if s.endswith("```"):
        s = s[:-3].rstrip()

    just = ""
    m = _JUST_RE.search(s)
    if m:
        just = m.group(1)

    last_close = s.rfind("}")
    if last_close > 0:
        try:
            obj = json.loads(s[: last_close + 1])
            if isinstance(obj, dict) and "score" in obj:
                score = _finite_score(obj["score"])
                if score is not None:
                    return (score, str(obj.get("justification", just)), True)
        except Exception:
            pass
    for m in _JSON_BLOCK_RE.finditer(s):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "score" in obj:
                score = _finite_score(obj["score"])
                if score is not None:
                    return (score, str(obj.get("justification", just)), True)
        except Exception:
            continue
    matches = list(_SCORE_RE.finditer(s))
    if matches:
        score = _finite_score(matches[-1].group(1))
        if score is not None:
            return score, just, False
    bare = list(_BARE_RE.finditer(s))
    if bare:
        score = _finite_score(bare[0].group(1))
        if score is not None:
            return score, just, False
    return None, just, False


def encode_prompt(processor, user_text: str, image,
                  exemplars: list[dict] | None = None,
                  batch: bool = False):
    """Tokenize ONE prompt (+ its images) with truncation DISABLED.

    Truncating a multimodal prompt silently cuts image placeholder tokens, so
    the text and input_ids disagree and processors raise
    "Mismatch in `image` token count between text and `input_ids`".
    Length is instead controlled by `fit_exemplars` (dropping exemplars) and
    by the per-model `max_image_long_side`.
    """
    prompt = apply_chat_template_safe(
        processor, user_text=user_text, exemplars=exemplars,
        add_generation_prompt=True)
    ex_images = [e["image"] for e in (exemplars or [])]
    # multi-image items must be nested one list per text
    images = [ex_images + [image]] if ex_images else [image]
    return processor(text=[prompt], images=images, return_tensors="pt",
                     padding=batch, truncation=False)


def fit_exemplars(processor, dataset, exemplars: list[dict],
                  max_length: int,
                  n_probe: int | None = None) -> list[dict] | None:
    """Largest k <= len(exemplars) whose prompt fits `max_length` tokens.

    Every extra exemplar adds a whole image, and models with dynamic tiling
    (InternVL3, Pixtral) turn one tall answer image into 1000+ tokens — 4
    images easily blow past a 4096-token budget. Rather than truncate (which
    corrupts the image tokens) we drop exemplars until the prompt fits.

    By default every evaluation item is checked. Dynamic image tiling means
    the first few records are not a safe proxy: a later tall/wide answer can
    contain thousands more image tokens. ``n_probe`` remains available only
    for explicit diagnostic/tests; production callers use the full split.
    """
    if not exemplars:
        return None
    n_items = len(dataset) if n_probe is None else min(int(n_probe),
                                                       len(dataset))
    if n_items < 1:
        return None
    scope = "entire split" if n_items == len(dataset) else f"{n_items} probes"
    print(f"  [few-shot] validating prompt budget against {scope}...")
    zero_shot_failure = None
    for k in range(len(exemplars), -1, -1):
        ex = exemplars[:k] or None
        longest = 0
        longest_idx = -1
        failed = False
        for i in range(n_items):
            item = dataset[i]
            try:
                enc = encode_prompt(processor, item["user_text"],
                                    item["image"], ex)
                length = int(enc["input_ids"].shape[1])
            except Exception as e:
                print(f"  [few-shot] k={k} failed to encode item {i}: {e}")
                failed = True
                if k == 0:
                    zero_shot_failure = f"item {i} failed to encode: {e}"
                break
            if length > longest:
                longest, longest_idx = length, i
            if longest > max_length:
                failed = True
                if k == 0:
                    zero_shot_failure = (
                        f"item {longest_idx} needs {longest} tokens")
                break
        if failed:
            if longest > max_length:
                print(f"  [few-shot] k={k}: item {longest_idx} needs "
                      f"{longest} tokens > max_length={max_length}")
            continue
        if k < len(exemplars):
            print(f"  [few-shot] reduced {len(exemplars)} -> {k} exemplars "
                  f"to fit max_length={max_length} across {n_items} items "
                  f"(longest prompt {longest} tokens at item {longest_idx})")
        else:
            print(f"  [few-shot] {k} exemplars fit across {n_items} items "
                  f"(longest {longest}/{max_length} tokens at item "
                  f"{longest_idx})")
        return ex
    raise ValueError(
        f"few-shot prompt cannot fit the configured max_length={max_length} "
        f"even after removing every exemplar ({zero_shot_failure}); reduce "
        f"models.<name>.max_image_long_side or raise few_shot.max_length")


class DonutInferenceCollator:
    """Spawn-safe Donut inference collator.

    Windows always starts DataLoader workers with ``spawn``.  A nested
    ``collate`` closure cannot be pickled by that start method, so collators
    used by workers must be module-level callables.
    """

    def __init__(self, processor, max_text: int, max_chars: int,
                 task_token: str):
        self.processor = processor
        self.max_text = int(max_text)
        self.max_chars = int(max_chars)
        self.task_token = str(task_token)

    def __call__(self, batch):
        processor = self.processor
        tok = processor.tokenizer
        tok.padding_side = "right"       # decoder prefix grows to the right
        images = [item["image"] for item in batch]
        prompts = [self.task_token + build_donut_prompt(
                   item["example"], self.max_chars)
                   for item in batch]
        pixel_values = processor(images, return_tensors="pt").pixel_values
        dec = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=self.max_text,
                  add_special_tokens=False)
        return {"inputs": {"pixel_values": pixel_values,
                           "decoder_input_ids": dec["input_ids"],
                           "decoder_attention_mask": dec["attention_mask"]},
                "examples": [item["example"] for item in batch]}


def make_donut_infer_collator(processor, cfg: Config):
    """Return a spawn-safe Donut inference collator."""
    return DonutInferenceCollator(
        processor,
        max_text=int(cfg.get_in("donut.max_text_length", 512)),
        max_chars=int(cfg.get_in("donut.max_prompt_chars", 1200)),
        task_token=str(cfg.get_in("donut.task_start_token", "<s_grade>")),
    )


class InferenceCollator:
    """Spawn-safe chat-VLM inference collator."""

    def __init__(self, processor, max_length: int = 4096,
                 exemplars: list[dict] | None = None):
        self.processor = processor
        self.max_length = int(max_length)
        self.exemplars = exemplars
        self.exemplar_images = [e["image"] for e in (exemplars or [])]

    def __call__(self, batch):
        processor = self.processor
        exemplars = self.exemplars
        ex_images = self.exemplar_images
        set_padding_side(processor, "left")
        prompts, images, examples = [], [], []
        for item in batch:
            prompts.append(apply_chat_template_safe(
                processor, user_text=item["user_text"],
                exemplars=exemplars, add_generation_prompt=True))
            if ex_images:
                images.append(ex_images + [item["image"]])
            else:
                images.append(item["image"])
            examples.append(item["example"])
        inp = processor(text=prompts, images=images, return_tensors="pt",
                        padding=True, truncation=False)
        if "attention_mask" in inp:
            longest = int(inp["attention_mask"].sum(dim=1).max().item())
        else:
            longest = int(inp["input_ids"].shape[1])
        if longest > self.max_length:
            raise ValueError(
                f"multimodal prompt has {longest} tokens, above the configured "
                f"budget {self.max_length}; reduce image size/exemplars or raise the "
                f"budget (prompts are never truncated)")
        return {"inputs": inp, "examples": examples}


def make_infer_collator(processor, max_length: int = 4096,
                        exemplars: list[dict] | None = None):
    """Left-pad a decoder-only batch without truncating image tokens.

    With few-shot prompting each item carries k+1 images in the nested shape
    expected by multi-image processors.  The returned module-level callable
    is picklable by Windows/Python ``spawn`` DataLoader workers.
    """
    return InferenceCollator(processor, max_length, exemplars)


def run_inference(cfg: Config, model_key: str, regime: str,
                  processor, model, test_ds, out_path: Path,
                  exemplars: list[dict] | None = None,
                  do_sample: bool | None = None,
                  temperature: float | None = None,
                  top_p: float | None = None,
                  seed: int | None = None) -> list[dict]:
    """Generate + parse scores for the whole test set; write JSONL."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    icfg = cfg.inference
    do_sample = icfg.do_sample if do_sample is None else do_sample
    temperature = icfg.temperature if temperature is None else temperature
    if seed is not None:
        torch.manual_seed(seed)

    donut = is_donut(cfg, model_key)

    # Few-shot: the prompt budget may be larger than the zero-shot one, and
    # the exemplar count is trimmed to whatever actually fits.
    budget = int(icfg.max_length)
    if exemplars and not donut:
        fs_max = cfg.get_in("few_shot.max_length")
        if fs_max:
            budget = int(fs_max)
        exemplars = fit_exemplars(processor, test_ds, exemplars, budget)
    elif exemplars and donut:
        # Donut is an encoder-decoder with a single image input and no
        # in-context-learning ability — exemplars cannot be supplied.
        print("  [donut] few-shot exemplars are not supported by an "
              "encoder-decoder model with one image input; running the same "
              "prompt as zero-shot (reported as few_shot for comparability)")
        exemplars = None

    bs = Config(cfg.models[model_key]).batch_size_infer
    if exemplars:  # k extra images per item
        bs = max(1, bs // (1 + len(exemplars)))

    print(f"\n=== Inference: {model_key} / {regime} "
          f"({len(test_ds)} examples, bs={bs}, sample={do_sample}, "
          f"exemplars={len(exemplars or [])}) ===")
    model.eval()
    if donut:
        collate = make_donut_infer_collator(processor, cfg)
    else:
        set_padding_side(processor, "left")
        collate = make_infer_collator(processor, budget, exemplars)
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                        collate_fn=collate,
                        num_workers=int(icfg.get("num_workers", 2)),
                        pin_memory=torch.cuda.is_available())

    tok = getattr(processor, "tokenizer", None)
    gen_kwargs = dict(max_new_tokens=icfg.max_new_tokens,
                      do_sample=do_sample, use_cache=True)
    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-3)
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    if tok is not None:
        if tok.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tok.pad_token_id
        if tok.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = tok.eos_token_id

    if donut:
        # encoder-decoder: the decoder prefix is not part of the "input_ids"
        # returned by generate, and Donut only needs a few tokens.
        gen_kwargs["max_new_tokens"] = int(cfg.get_in("donut.max_new_tokens",
                                                      16))
        gen_kwargs.pop("use_cache", None)

    clamp_preds = bool(icfg.get("clamp_to_criterion_max", True))
    rows = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{model_key}/{regime}"):
            inp = move_inputs_to_model(batch["inputs"], model)
            out_ids = model.generate(**inp, **gen_kwargs)
            if donut:
                # strip the decoder prefix we supplied
                in_len = inp["decoder_input_ids"].shape[1]
                texts = processor.tokenizer.batch_decode(
                    out_ids[:, in_len:], skip_special_tokens=True)
            else:
                in_len = inp["input_ids"].shape[1]
                texts = processor.batch_decode(out_ids[:, in_len:],
                                               skip_special_tokens=True)
            for ex, raw in zip(batch["examples"], texts):
                pred, just, ok = parse_score(raw, allow_bare=donut)
                mx = float(ex.get("resolved_max") or ex.get("criterion_max") or 0)
                raw_pred = pred
                if pred is not None and clamp_preds:
                    pred = clamp(pred, mx)
                rows.append({
                    "example_id": ex["example_id"],
                    "answer_id": ex["answer_id"],
                    "criterion_name": ex["criterion_name"],
                    "criterion_index": ex["criterion_index"],
                    "model": model_key,
                    "regime": regime,
                    "pred_score": pred,
                    "pred_score_raw": raw_pred,
                    "true_score": float(ex["gained_marks"]),
                    "max_score": mx,
                    "criterion_max_dataset": float(ex.get("criterion_max") or 0),
                    "justification": just,
                    "raw_output": raw.strip(),
                    "parse_ok": ok,
                    "n_exemplars": len(exemplars or []),
                })
    n_ok = sum(1 for r in rows if r["pred_score"] is not None)
    print(f"  parsed {n_ok}/{len(rows)} ({n_ok / max(1, len(rows)):.1%})")
    write_records(rows, out_path)
    return rows
