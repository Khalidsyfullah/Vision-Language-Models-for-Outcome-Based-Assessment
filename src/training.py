"""Training for the `lora` and resource-aware `full` regimes.

  lora  parameter-efficient: base weights frozen, low-rank adapters trained.
        Works for 7-12B models on one GPU, optionally over 4-bit weights.

  full  Donut updates 100% of its ~200M parameters. The three 7-12B chat VLMs
        update a suffix of complete top language blocks nearest the configured
        30% of all parameters, with vision/projector/embeddings/lower blocks
        frozen. The selected memory footprint is checked before training.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from .config import Config
from .models import (count_parameters, estimate_selected_finetune_gb,
                     gpu_memory_gb, is_donut, model_float_dtype,
                     processor_accepts, set_padding_side)
from .prompting import (apply_chat_template_safe, build_donut_prompt,
                        donut_target, target_json)


# --------------------------------------------------------------------------
# Collators
# --------------------------------------------------------------------------

def _flat_token_ids(encoded) -> list[int]:
    """Extract one unbatched token-id list from a tokenizer result."""
    ids = (encoded.get("input_ids") if isinstance(encoded, dict)
           else getattr(encoded, "input_ids", encoded))
    if ids is None:
        return []
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(token) for token in ids]


def _rfind_tokens(sequence: list[int], target: list[int]) -> int | None:
    """Return the rightmost start of ``target`` in ``sequence``."""
    if not target or len(target) > len(sequence):
        return None
    for start in range(len(sequence) - len(target), -1, -1):
        if sequence[start:start + len(target)] == target:
            return start
    return None


def _flat_offsets(encoded) -> list[tuple[int, int]]:
    """Extract one unbatched fast-tokenizer offset mapping."""
    offsets = (encoded.get("offset_mapping") if isinstance(encoded, dict)
               else getattr(encoded, "offset_mapping", None))
    if offsets is None:
        return []
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    return [(int(start), int(end)) for start, end in offsets]


def _token_id(tokenizer, token: str) -> int | None:
    """Resolve a real special token ID without accepting an unknown ID."""
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if isinstance(token_id, (list, tuple)):
        if len(token_id) != 1:
            return None
        token_id = token_id[0]
    try:
        token_id = int(token_id)
    except (TypeError, ValueError):
        return None
    unknown = getattr(tokenizer, "unk_token_id", None)
    if token_id < 0 or (unknown is not None and token_id == int(unknown)):
        return None
    return token_id


def _chatml_assistant_start(tokenizer, expanded_ids) -> int | None:
    """Find the final ChatML assistant turn using structural delimiters.

    InternVL3's official template is ``<|im_start|>role\ncontent<|im_end|>``.
    Image tiling can only change tokens inside an earlier user turn, so the
    final assistant delimiter and role are invariant across every record.  We
    deliberately skip one token after the role; it is the newline, or a token
    that merged the newline with the first answer character. In the latter
    case the first answer token remains masked, which is safe.
    """
    if not callable(tokenizer):
        return None
    start_id = _token_id(tokenizer, "<|im_start|>")
    end_id = _token_id(tokenizer, "<|im_end|>")
    if start_id is None or end_id is None:
        return None
    sequence = [int(token) for token in expanded_ids]
    starts = [i for i, token in enumerate(sequence) if token == start_id]
    if not starts:
        return None
    turn_start = starts[-1]
    try:
        turn_end = sequence.index(end_id, turn_start + 1)
    except ValueError:
        return None

    # Decode only the tiny structural header. This is robust when BPE stores
    # ``assistant\n`` (or even ``assistant\n{``) as a merged token: once the
    # decoded prefix reaches the header, that whole final token remains masked.
    decode_limit = min(turn_end, turn_start + 12)
    for stop in range(turn_start + 2, decode_limit + 1):
        try:
            decoded = tokenizer.decode(
                sequence[turn_start + 1:stop],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
        except TypeError:
            try:
                decoded = tokenizer.decode(
                    sequence[turn_start + 1:stop],
                    skip_special_tokens=False)
            except Exception:
                break
        except Exception:
            break
        if str(decoded).startswith("assistant\n"):
            return stop if stop + 2 <= turn_end else None

    # Older/minimal tokenizers may not expose decode. Fall back to matching the
    # role tokens and conservatively skipping one newline/context token.
    try:
        role_ids = _flat_token_ids(tokenizer(
            "assistant", add_special_tokens=False))
    except Exception:
        return None
    if not role_ids:
        return None

    # The role normally begins immediately after <|im_start|>. Search a small
    # structural window for tokenizer-version compatibility, never the whole
    # answer where the word "assistant" might occur naturally.
    role_start = None
    upper = min(turn_end, turn_start + 1 + 6)
    for index in range(turn_start + 1, upper):
        if sequence[index:index + len(role_ids)] == role_ids:
            role_start = index
            break
    if role_start is None:
        return None
    boundary = role_start + len(role_ids) + 1
    # Require at least two answer/control tokens before the matching turn end.
    return boundary if boundary + 2 <= turn_end else None


def _rendered_assistant_start(tokenizer, rendered_text: str,
                              assistant_text: str, expanded_ids):
    """Map the answer's character boundary through image-token expansion.

    The tokenizer is applied to the *complete rendered chat*, preserving both
    left and right context around the JSON target. Its post-boundary token tail
    must then occur intact near the end of the multimodal processor sequence;
    image expansion only changes the earlier prefix. This avoids assuming that
    a standalone target tokenizes identically inside a chat template.
    """
    if not callable(tokenizer):
        return None
    char_start = rendered_text.rfind(assistant_text)
    if char_start < 0:
        return None
    try:
        encoded = tokenizer(rendered_text, add_special_tokens=False,
                            return_offsets_mapping=True)
    except Exception:
        try:
            encoded = tokenizer(rendered_text, add_special_tokens=False)
        except Exception:
            return None
    raw_ids = _flat_token_ids(encoded)
    offsets = _flat_offsets(encoded)
    if not raw_ids:
        return None

    raw_boundary = None
    if len(offsets) == len(raw_ids):
        for index, (start, end) in enumerate(offsets):
            if end <= char_start:
                continue
            # A token spanning the prompt/answer character boundary is unsafe
            # to supervise; leave it masked and begin with the next.
            raw_boundary = index + (1 if start < char_start else 0)
            break
    else:
        # InternVL3-8B-hf declares Qwen2Tokenizer (the slow tokenizer), which
        # cannot return character offsets. Tokenizing the exact rendered prefix
        # gives a conservative boundary: if its final token would merge with
        # the answer in full context, starting at this token count masks that
        # merged first-answer token rather than leaking a prompt token.
        try:
            prefix_ids = _flat_token_ids(tokenizer(
                rendered_text[:char_start], add_special_tokens=False))
            raw_boundary = len(prefix_ids)
        except Exception:
            return None
    if raw_boundary is None or raw_boundary >= len(raw_ids):
        return None

    expanded = [int(token) for token in expanded_ids]
    # A processor can append one or more automatic special tokens. Search for
    # the rightmost complete rendered tail instead of requiring it to be the
    # literal final slice. Dropping up to two leading tail tokens remains safe
    # and handles a context-merged first answer token.
    for dropped in range(5):
        tail = raw_ids[raw_boundary + dropped:]
        if len(tail) < 2:
            break
        start = _rfind_tokens(expanded, tail)
        if start is not None:
            return start
    return None


def _assistant_token_start(tokenizer, full_ids, assistant_text: str):
    """Locate a rendered assistant target without reprocessing the image.

    Tokenizers can merge the first target token with template whitespace, so
    several whitespace contexts are tried.  As a final safe recovery, at most
    the first two target tokens may be left masked when the remaining suffix
    matches exactly.  This can omit a tiny part of the supervised target but
    can never train on prompt/image tokens.
    """
    if not callable(tokenizer):
        return None
    sequence = [int(token) for token in full_ids]
    candidates: list[list[int]] = []
    for prefix in ("", " ", "\n", "\n\n"):
        try:
            ids = _flat_token_ids(tokenizer(
                prefix + assistant_text, add_special_tokens=False))
        except Exception:
            continue
        if ids and ids not in candidates:
            candidates.append(ids)

    exact = [_rfind_tokens(sequence, ids) for ids in candidates]
    exact = [start for start in exact if start is not None]
    if exact:
        return max(exact)

    # Context-sensitive BPE/SentencePiece tokenization normally changes only
    # the first token. Permit two for defensive compatibility, but require at
    # least two exact supervised tokens to remain.
    suffix_matches: list[tuple[int, int]] = []
    for ids in candidates:
        for dropped in range(1, min(2, len(ids) - 2) + 1):
            suffix = ids[dropped:]
            start = _rfind_tokens(sequence, suffix)
            if start is not None:
                suffix_matches.append((len(suffix), start))
    if suffix_matches:
        # Prefer the longest exact suffix, then its rightmost occurrence.
        return max(suffix_matches)[1]
    return None


def encode_training_batch(processor, texts, images, side: str = "right"):
    """Encode one chat-VLM training batch with an explicitly pinned pad side.

    `set_padding_side` only writes `processor.tokenizer.padding_side`, which a
    multimodal processor may override. `InternVLProcessorKwargs` declares
    ``_defaults["text_kwargs"]["padding_side"] = "left"``, and
    `ProcessorMixin._merge_kwargs` only lets the tokenizer attribute win when
    the checkpoint's own ``tokenizer_config.json`` lists ``padding_side``.
    InternVL3-8B-hf's does not, so its *training* batches came back LEFT
    padded while every other model here was right padded. The flat keyword
    below has the highest merge priority, so the side is now pinned for real.
    """
    set_padding_side(processor, side)
    kwargs = dict(text=texts, images=images, return_tensors="pt",
                  padding=True, truncation=False)
    if processor_accepts(processor, "padding_side"):
        try:
            return processor(**kwargs, padding_side=side)
        except (TypeError, ValueError):
            # A processor version that does not route the keyword. The mask
            # span below keeps labels correct for either side anyway.
            pass
    return processor(**kwargs)


def real_token_span(mask_row) -> tuple[int, int]:
    """Return ``(start, end)`` of the unpadded region of an attention mask.

    Reading the mask instead of assuming ``[0:mask.sum()]`` makes every
    boundary search and label write correct under left, right, or absent
    padding. The previous slice silently returned ``pad * k + prefix`` for a
    left-padded row, cutting the assistant turn off the end.
    """
    row = mask_row.tolist() if hasattr(mask_row, "tolist") else list(mask_row)
    values = [int(value) for value in row]
    nonzero = [index for index, value in enumerate(values) if value != 0]
    if not nonzero:
        return 0, len(values)
    return nonzero[0], nonzero[-1] + 1


class TrainingCollator:
    """Spawn-safe chat-VLM training collator."""

    def __init__(self, processor, max_length: int = 4096, float_dtype=None):
        self.processor = processor
        self.max_length = int(max_length)
        self.float_dtype = float_dtype

    def __call__(self, batch):
        import torch

        processor = self.processor
        full_texts, user_texts, assistant_texts, images = [], [], [], []
        for item in batch:
            ut, at = item["user_text"], item["assistant_text"]
            full_texts.append(apply_chat_template_safe(
                processor, user_text=ut, assistant_text=at))
            user_texts.append(ut)
            assistant_texts.append(at)
            images.append(item["image"])
        # Multimodal processors expand one textual image placeholder into
        # hundreds/thousands of input IDs. Tokenizer-only lengths are
        # therefore wrong. Encode each completed example once. ChatML models
        # use the final assistant-turn delimiter; other templates use the
        # conservative context-preserving fallbacks below. One pass also avoids
        # dynamic image tiling changing separately processed prefixes.
        inp = encode_training_batch(processor, full_texts, images)
        seq = inp["input_ids"].shape[1]
        if seq > self.max_length:
            raise ValueError(
                f"training sequence has {seq} tokens, above "
                f"training.max_length={self.max_length}. Lower the model image size "
                f"or raise the configured budget; multimodal prompts cannot be "
                f"safely truncated.")
        if self.float_dtype is not None:
            for key, value in list(inp.items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    inp[key] = value.to(dtype=self.float_dtype)

        labels = inp["input_ids"].clone()
        unresolved = []
        for i, answer in enumerate(assistant_texts):
            row = inp["input_ids"][i].detach().cpu()
            # Boundaries are searched inside the row's REAL tokens, and the
            # resulting index is then shifted back by the row's pad offset.
            if "attention_mask" in inp:
                start, end = real_token_span(inp["attention_mask"][i])
            else:
                start, end = 0, int(row.numel())
            full = row[start:end]
            boundary = _chatml_assistant_start(
                processor.tokenizer, full.tolist())
            if boundary is None:
                boundary = _rendered_assistant_start(
                    processor.tokenizer, full_texts[i], answer, full.tolist())
            if boundary is None:
                boundary = _assistant_token_start(
                    processor.tokenizer, full.tolist(), answer)
            if boundary is None:
                unresolved.append((i, full, start))
            else:
                labels[i, :start + boundary] = -100

        # Conservative compatibility fallback for unusual tokenizers that do
        # not expose callable text tokenization. Only unresolved rows pay for
        # a second multimodal processor pass.
        if unresolved:
            indexes = [i for i, _, _ in unresolved]
            boundary_texts = [apply_chat_template_safe(
                processor, user_text=user_texts[i],
                assistant_text="OBE_BOUNDARY_SENTINEL_DO_NOT_TRAIN")
                              for i in indexes]
            boundary_inp = encode_training_batch(
                processor, boundary_texts, [images[i] for i in indexes])
            for fallback_i, (batch_i, full, start) in enumerate(unresolved):
                bids = boundary_inp["input_ids"][fallback_i].detach().cpu()
                # Both passes pad independently, so each row is trimmed with
                # its own mask span before the prefixes are compared.
                if "attention_mask" in boundary_inp:
                    b_start, b_end = real_token_span(
                        boundary_inp["attention_mask"][fallback_i])
                    bids = bids[b_start:b_end]
                width = min(int(full.numel()), int(bids.numel()))
                common = 0
                while (common < width and
                       int(full[common]) == int(bids[common])):
                    common += 1
                if common < 1 or common >= int(full.numel()):
                    example = batch[batch_i].get("example", {})
                    example_id = example.get("example_id", batch_i)
                    raise ValueError(
                        "could not safely locate the assistant-answer boundary "
                        f"for training example {example_id!r} "
                        f"(full_tokens={full.numel()}, "
                        f"sentinel_tokens={bids.numel()}, "
                        f"common_prefix={common}, pad_offset={start})")
                labels[batch_i, :start + common] = -100
            del boundary_inp
        if "attention_mask" in inp:
            labels[inp["attention_mask"] == 0] = -100
        supervised = (labels != -100).sum(dim=1)
        if int(supervised.min().item()) < 1:
            blank = int(supervised.argmin().item())
            example = batch[blank].get("example", {})
            raise ValueError(
                "training example "
                f"{example.get('example_id', blank)!r} has no supervised "
                "answer token after prompt masking")
        inp["labels"] = labels
        # Trainer evaluation runs through the same collator. Passing this
        # explicitly prevents ModelOutput.past_key_values from containing a
        # Transformers Cache that native-Windows DataParallel cannot gather.
        inp["use_cache"] = False
        return inp


def make_train_collator(processor, max_length: int = 4096,
                        float_dtype=None):
    """Return a spawn-safe chat-VLM collator.

    Right padding is used and prompt tokens are masked to -100, so loss covers
    only the gold JSON answer. Truncation remains disabled because cutting a
    multimodal prompt removes expanded image placeholder tokens.
    """
    return TrainingCollator(processor, max_length, float_dtype)


class DonutTrainingCollator:
    """Spawn-safe Donut training collator."""

    def __init__(self, processor, max_text: int, max_chars: int,
                 task_token: str, float_dtype=None):
        self.processor = processor
        self.max_text = int(max_text)
        self.max_chars = int(max_chars)
        self.task_token = str(task_token)
        self.float_dtype = float_dtype

    def __call__(self, batch):
        import torch

        processor = self.processor
        tok = processor.tokenizer
        images = [item["image"] for item in batch]
        pixel_values = processor(images, return_tensors="pt").pixel_values
        if self.float_dtype is not None:
            pixel_values = pixel_values.to(dtype=self.float_dtype)

        seqs, prompt_lens = [], []
        for item in batch:
            ex = item["example"]
            prompt = self.task_token + build_donut_prompt(ex, self.max_chars)
            p_ids = tok(prompt, add_special_tokens=False,
                        truncation=True, max_length=self.max_text).input_ids
            t_ids = tok(donut_target(ex),
                        add_special_tokens=False).input_ids + [tok.eos_token_id]
            seqs.append(p_ids + t_ids)
            prompt_lens.append(len(p_ids))

        width = max(len(s) for s in seqs)
        pad = tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for s, plen in zip(seqs, prompt_lens):
            padding = [pad] * (width - len(s))
            ids = s + padding
            lab = list(s) + [-100] * len(padding)
            lab[:plen] = [-100] * plen          # don't learn to echo the prompt
            input_ids.append(ids)
            labels.append(lab)
            attn.append([1] * len(s) + [0] * len(padding))

        ids = torch.tensor(input_ids)
        lab = torch.tensor(labels)
        # teacher forcing: decoder reads position t, predicts position t+1
        return {"pixel_values": pixel_values,
                "decoder_input_ids": ids[:, :-1],
                "decoder_attention_mask": torch.tensor(attn)[:, :-1],
                "labels": lab[:, 1:],
                # VisionEncoderDecoderModel otherwise returns its decoder
                # Cache during epoch evaluation, which Windows DataParallel
                # cannot gather when it includes lazy None entries.
                "use_cache": False}


def make_donut_train_collator(processor, cfg: Config, float_dtype=None):
    """Return a spawn-safe Donut collator.

    The image is sent to the encoder; the task token, prompt, target, and EOS
    form the decoder sequence. Prompt positions are masked from the loss.
    """
    return DonutTrainingCollator(
        processor,
        max_text=int(cfg.get_in("donut.max_text_length", 512)),
        max_chars=int(cfg.get_in("donut.max_prompt_chars", 1200)),
        task_token=str(cfg.get_in("donut.task_start_token", "<s_grade>")),
        float_dtype=float_dtype,
    )


def _preflight_internvl_boundaries(processor, datasets) -> int:
    """Validate every train/validation target against InternVL ChatML.

    This is text-only: it reads score/id columns directly from the wrapped HF
    datasets, so it neither decodes images nor duplicates dynamic tiling. The
    structural rule is target-independent, but checking every row also rejects
    an empty/malformed target before the expensive Trainer loop starts.
    """
    checked = 0
    for dataset in datasets:
        raw = getattr(dataset, "ds", None)
        columns = set(getattr(raw, "column_names", []) or [])
        if raw is None or "gained_marks" not in columns:
            continue
        scores = raw["gained_marks"]
        ids = (raw["example_id"] if "example_id" in columns
               else list(range(len(scores))))
        for example_id, score in zip(ids, scores):
            answer = target_json({"gained_marks": score})
            rendered = apply_chat_template_safe(
                processor, user_text="InternVL boundary preflight.",
                assistant_text=answer)
            token_ids = _flat_token_ids(processor.tokenizer(
                rendered, add_special_tokens=False))
            boundary = _chatml_assistant_start(
                processor.tokenizer, token_ids)
            if boundary is None:
                raise ValueError(
                    "InternVL ChatML preflight could not locate the final "
                    f"assistant turn for example {example_id!r}")
            checked += 1
    return checked


def _preflight_supervised_labels(collator, processor, dataset,
                                 rows: int = 4) -> dict:
    """Collate a real multi-row batch and verify what the loss will see.

    Run in the main process before DataLoader workers are spawned. Using more
    than one row is the point: a single-row batch never pads, so it cannot
    detect a processor that pads on the unexpected side, which is exactly how
    InternVL3 reached the Trainer loop with unusable label offsets. Every
    supervised span must decode back inside that row's gold answer, so a
    prompt or image token can never silently enter the loss.
    """
    count = min(len(dataset), max(2, int(rows)))
    items = [dataset[i] for i in range(count)]
    probe = collator(items)
    labels, ids = probe["labels"], probe["input_ids"]
    tokenizer = getattr(processor, "tokenizer", None)
    padded = False
    if "attention_mask" in probe:
        mask = probe["attention_mask"]
        padded = bool((mask == 0).any().item())
        sides = {real_token_span(mask[i])[0] for i in range(labels.shape[0])}
        if padded and sides != {0}:
            # Labels stay correct either way (spans are read from the mask),
            # but right padding keeps train-time positions identical to
            # inference, so an unexpected side is still worth reporting.
            print(f"    [warn] this processor pads on the left "
                  f"(pad offsets {sorted(sides)}) despite the pinned side")
    totals = []
    for row in range(labels.shape[0]):
        keep = labels[row] != -100
        supervised = int(keep.sum().item())
        if supervised < 2:
            raise ValueError(
                f"row {row} of the collated probe has {supervised} supervised "
                "tokens; the assistant answer was not located")
        totals.append(supervised)
        if tokenizer is None or not hasattr(tokenizer, "decode"):
            continue
        try:
            text = tokenizer.decode(ids[row][keep].tolist(),
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
        except TypeError:
            try:
                text = tokenizer.decode(ids[row][keep].tolist(),
                                        skip_special_tokens=True)
            except Exception:
                continue
        except Exception:
            continue
        text = str(text).strip()
        if text and text not in items[row]["assistant_text"]:
            raise ValueError(
                "supervised tokens fall outside the gold answer for row "
                f"{row}: masked-in text {text[:120]!r} is not part of "
                f"{items[row]['assistant_text'][:120]!r}")
    del probe
    return {"rows": len(totals), "padded": padded,
            "supervised": totals}


# --------------------------------------------------------------------------
# Resource-aware fine-tuning setup and feasibility check
# --------------------------------------------------------------------------


_VISION_NAME_PARTS = ("vision", "visual", "vit", "patch_embed",
                      "image_encoder")
_LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)")


def _is_vision_parameter(name: str) -> bool:
    lower = name.lower()
    return any(part in lower for part in _VISION_NAME_PARTS)


def unfreeze_top_language_fraction(model, fraction: float) -> dict:
    """Freeze the model, then train a top-layer suffix nearest ``fraction``.

    Whole transformer blocks are selected so the strategy remains meaningful
    and reproducible; individual tensors are never cut in half. Vision-side
    parameters and multimodal projectors remain frozen. The final language
    norm/output head is included when it is not tied to the input embedding.
    """
    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError("training.full.chat_unfreeze_fraction must be in (0, 1]")

    named = list(model.named_parameters())
    total = sum(p.numel() for _, p in named)
    if fraction == 1:
        for _, parameter in named:
            parameter.requires_grad = True
        return {"strategy": "all_parameters", "requested_fraction": 1.0,
                "selected_layers": [], "trainable_parameters": total,
                "total_parameters": total, "actual_fraction": 1.0}

    for _, parameter in named:
        parameter.requires_grad = False

    layer_parameters: dict[int, list] = {}
    output_parameters = []
    for name, parameter in named:
        lower = name.lower()
        if _is_vision_parameter(lower):
            continue
        matches = _LAYER_PATTERN.findall(lower)
        if matches:
            layer_parameters.setdefault(int(matches[-1]), []).append(parameter)
            continue
        if ("lm_head" in lower or "output_projection" in lower or
                "final_layer_norm" in lower or
                lower.endswith((".norm.weight", ".norm.bias"))):
            output_parameters.append(parameter)

    if not layer_parameters:
        preview = ", ".join(name for name, _ in named[:8])
        raise RuntimeError(
            "could not identify language transformer layers for partial "
            f"fine-tuning; first parameter names: {preview}")

    fixed = sum(p.numel() for p in output_parameters)
    target = total * fraction
    descending = sorted(layer_parameters, reverse=True)
    cumulative = fixed
    candidates = []
    for index in descending:
        cumulative += sum(p.numel() for p in layer_parameters[index])
        candidates.append((abs(cumulative - target), index, cumulative))
    # Pick the number of complete top layers closest to the target. At least
    # one layer is always trained, even for a very small configured fraction.
    best_position = min(range(len(candidates)),
                        key=lambda i: candidates[i][0])
    selected = descending[:best_position + 1]
    for parameter in output_parameters:
        parameter.requires_grad = True
    for index in selected:
        for parameter in layer_parameters[index]:
            parameter.requires_grad = True

    trainable, _ = count_parameters(model)
    return {
        "strategy": "top_language_layers",
        "requested_fraction": fraction,
        "selected_layers": sorted(selected),
        "available_language_layers": sorted(layer_parameters),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "actual_fraction": trainable / max(1, total),
    }


def check_finetune_feasible(model, model_key: str, strategy: str,
                            allow_anyway: bool = False) -> None:
    need = estimate_selected_finetune_gb(model)
    have = gpu_memory_gb()
    trainable, total = count_parameters(model)
    print(f"  {strategy}: {trainable / 1e6:.0f}M / "
          f"{total / 1e6:.0f}M parameters trainable "
          f"({100 * trainable / max(1, total):.1f}%), "
          f"~{need:.1f} GB parameter/optimizer memory, "
          f"{have:.1f} GB available")
    if have and need > have * 0.9 and not allow_anyway:
        raise SystemExit(
            f"\nThe configured {strategy} for {model_key} needs roughly "
            f"{need:.0f} GB before activations but "
            f"this GPU has {have:.0f} GB.\n"
            f"Options:\n"
            f"  * lower training.full.chat_unfreeze_fraction\n"
            f"  * use --regime lora\n"
            f"  * use Linux FSDP / DeepSpeed ZeRO-3 with CPU offload\n"
            f"  * set training.full.force: true to try anyway")


def _set_model_use_cache(model, enabled: bool) -> int:
    """Set every reachable HF config's cache flag for train/eval safety.

    Native-Windows DataParallel tries to gather every ModelOutput field.
    Transformer Cache objects are not gatherable and may contain lazy ``None``
    entries, which caused Donut evaluation to fail after its first epoch.
    Teacher-forced training/evaluation never needs KV caching.
    """
    pending, seen, changed = [model], set(), 0
    while pending:
        obj = pending.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        config = getattr(obj, "config", None)
        if config is not None:
            if hasattr(config, "use_cache"):
                config.use_cache = bool(enabled)
                changed += 1
            decoder_config = getattr(config, "decoder", None)
            if decoder_config is not None and hasattr(
                    decoder_config, "use_cache"):
                decoder_config.use_cache = bool(enabled)
                changed += 1
        for attr in ("base_model", "model", "decoder", "language_model"):
            try:
                child = getattr(obj, attr, None)
            except Exception:
                child = None
            if child is not None and child is not obj:
                pending.append(child)
    return changed


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------

def _training_arguments(**kwargs):
    """TrainingArguments compatible across transformers versions.

    `evaluation_strategy` was renamed to `eval_strategy` in 4.41; unknown
    keys are dropped rather than raising.
    """
    from transformers import TrainingArguments

    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" not in valid and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    dropped = [k for k in kwargs if k not in valid]
    for k in dropped:
        kwargs.pop(k)
    if dropped:
        print(f"    [warn] TrainingArguments ignored: {dropped}")
    return TrainingArguments(**kwargs)


def fine_tune(cfg: Config, model_key: str, processor, model,
              train_ds, val_ds, output_dir: str | Path,
              regime: str = "lora"):
    """Train `model` under the given regime and save it to `output_dir`."""
    import torch
    from transformers import Trainer

    tcfg = cfg.training
    mcfg = Config(cfg.models[model_key])
    donut = is_donut(cfg, model_key)
    if regime == "full":
        # Partial chat-VLM tuning has optimizer state for billions of
        # parameters. Conservative per-device batches leave room for
        # activations on 96 GB native-Windows GPUs; accumulation preserves the
        # requested effective batch size.
        bs = int(mcfg.get("batch_size_full", mcfg.batch_size_train))
        ga = int(mcfg.get("grad_accum_full", mcfg.grad_accum))
    else:
        bs, ga = int(mcfg.batch_size_train), int(mcfg.grad_accum)

    print(f"\n=== {regime} training: {model_key} ===")
    print(f"  train={len(train_ds)} val={len(val_ds)} "
          f"epochs={tcfg.epochs} eff_bs={bs * ga}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if regime == "lora":
        from peft import LoraConfig, get_peft_model

        if getattr(model, "is_loaded_in_4bit", False) or getattr(
                model, "is_loaded_in_8bit", False):
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=tcfg.gradient_checkpointing)

        targets = list(mcfg.get("lora_target_modules")
                       or tcfg.lora.target_modules)
        model = get_peft_model(model, LoraConfig(
            r=tcfg.lora.r, lora_alpha=tcfg.lora.alpha,
            lora_dropout=tcfg.lora.dropout, target_modules=targets,
            bias="none",
            # PeftModelForSeq2SeqLM injects input_ids into the encoder. A
            # VisionEncoderDecoderModel uses pixel_values instead, so Donut
            # must use the generic PeftModel wrapper.
            task_type=None if donut else "CAUSAL_LM",
        ))
        if tcfg.freeze_vision_tower and not donut:
            # Freeze after adapter injection: PEFT marks every inserted LoRA
            # parameter trainable, including adapters under a vision module.
            frozen = 0
            for n, p in model.named_parameters():
                if any(k in n.lower() for k in ("vision", "visual", "vit",
                                                "patch_embed",
                                                "image_encoder")):
                    p.requires_grad = False
                    frozen += p.numel()
            print(f"  vision tower frozen ({frozen / 1e6:.0f}M params)")
        model.print_trainable_parameters()
        lr = float(tcfg.learning_rate)
    elif regime == "full":
        fcfg = Config(tcfg.get("full", {}) or {})
        if donut:
            for parameter in model.parameters():
                parameter.requires_grad = True
            trainable, total = count_parameters(model)
            tune_info = {
                "strategy": "all_parameters",
                "requested_fraction": 1.0,
                "actual_fraction": trainable / max(1, total),
                "selected_layers": [],
                "trainable_parameters": trainable,
                "total_parameters": total,
            }
            strategy_label = "Donut full fine-tune"
        else:
            tune_info = unfreeze_top_language_fraction(
                model, float(fcfg.get("chat_unfreeze_fraction", 0.30)))
            strategy_label = "chat-VLM top-layer partial fine-tune"
            layers = tune_info["selected_layers"]
            print(f"  selected top language layers: {layers[0]}..{layers[-1]} "
                  f"({len(layers)} layers)")
            print("  vision tower, projector, embeddings, and lower language "
                  "layers frozen")
        check_finetune_feasible(model, model_key, strategy_label,
                                bool(fcfg.get("force", False)))
        lr = float(fcfg.get("learning_rate", 2e-5))
    else:
        raise ValueError(f"regime '{regime}' does not train weights")

    print(f"  learning rate: {lr}")
    cache_configs = _set_model_use_cache(model, False)
    if cache_configs:
        print(f"  KV cache disabled for train/eval ({cache_configs} configs)")
    if tcfg.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        except Exception as e:
            print(f"    [warn] gradient checkpointing unavailable: {e}")

    bf16_ok = (bool(tcfg.bf16) and torch.cuda.is_available()
               and torch.cuda.is_bf16_supported())
    args = _training_arguments(
        output_dir=str(output_dir),
        num_train_epochs=float(tcfg.epochs),
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=max(1, bs * 2),
        gradient_accumulation_steps=ga,
        learning_rate=lr,
        warmup_ratio=float(tcfg.warmup_ratio),
        weight_decay=float(tcfg.weight_decay),
        lr_scheduler_type=tcfg.lr_scheduler,
        max_grad_norm=float(tcfg.max_grad_norm),
        bf16=bf16_ok, fp16=(not bf16_ok and torch.cuda.is_available()),
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=bool(tcfg.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 2)),
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False, report_to="none",
        seed=int(cfg.seed),
        optim=tcfg.get("optim", "adamw_torch_fused"),
    )

    float_dtype = model_float_dtype(model)
    collator = (make_donut_train_collator(processor, cfg, float_dtype)
                if donut else make_train_collator(
                    processor, tcfg.max_length, float_dtype))
    if not donut:
        if model_key == "internvl3":
            checked = _preflight_internvl_boundaries(
                processor, (train_ds, val_ds))
            print(f"  InternVL ChatML preflight: {checked} text targets "
                  f"passed")
        # Exercise the real multimodal processor, its padding side, and the
        # label mask on a multi-row batch before Trainer/DataLoader startup.
        report = _preflight_supervised_labels(collator, processor, train_ds)
        print(f"  label preflight: {report['rows']} collated rows "
              f"({'padded' if report['padded'] else 'equal length'}), "
              f"supervised tokens {report['supervised']}")
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = None
    if regime == "full":
        metadata = {
            "model": model_key,
            "regime": regime,
            **tune_info,
            "batch_size_per_device": bs,
            "gradient_accumulation_steps": ga,
            "learning_rate": lr,
        }
        # Write provenance before training as well as after it. If a run is
        # interrupted after an epoch checkpoint, its partial/full strategy is
        # still explicit next to the recoverable Trainer checkpoint.
        (out / "obe_training_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
    trainer.train()

    # Generation benefits from caching, and inference passes use_cache=True.
    # Restore before saving so reloaded checkpoints have normal generation
    # defaults while every Trainer forward above remained cache-free.
    _set_model_use_cache(model, True)
    model.save_pretrained(out)     # LoRA -> adapter only; full -> all weights
    processor.save_pretrained(out)
    if metadata is not None:
        (out / "obe_training_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  checkpoint saved -> {out}")
    return model
