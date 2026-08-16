"""Prompt construction for all three regimes.

zero_shot : system + user(image, rubric)
few_shot  : system + k exemplar (user, assistant) turns with images + query
lora      : same as zero_shot at inference; training masks the prompt tokens

`apply_chat_template_safe` handles processors whose chat template rejects a
system role (Pixtral/Llava-style templates) by folding the system prompt into
the first user turn.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are an expert university exam grader. You will see a student's "
    "handwritten answer (image) along with the question, model answer, "
    "and rubric for ONE specific criterion. Grade ONLY that criterion. "
    'Respond with EXACTLY one JSON object on a single line: '
    '{"score": <number>, "justification": "<one short sentence>"}. '
    "Do not output code fences, markdown, or any other text."
)


def criterion_max_of(ex: dict) -> float:
    """Prefer the repaired maximum (see src/scoring.resolve_criterion_max)."""
    v = ex.get("resolved_max", ex.get("criterion_max"))
    return float(v) if v is not None else 0.0


def build_user_text(ex: dict) -> str:
    try:
        levels = json.loads(ex["rubric_levels"])
    except Exception:
        levels = {}
    levels_str = "\n".join(f"  - {k}: {v}" for k, v in levels.items())
    return (f"QUESTION:\n{ex['question']}\n\n"
            f"MODEL ANSWER (reference):\n{ex['model_answer']}\n\n"
            f"CRITERION TO GRADE: {ex['criterion_name']} "
            f"(max {criterion_max_of(ex):g} marks)\n"
            f"RUBRIC LEVELS:\n{levels_str}\n\n"
            f"Grade only this criterion based on the student's "
            f"handwritten answer in the image. The score must be between 0 "
            f"and {criterion_max_of(ex):g}. "
            f"Output a single JSON object only.")


def target_json(ex: dict) -> str:
    """Gold assistant turn used for LoRA training and few-shot exemplars."""
    return json.dumps(
        {"score": float(ex["gained_marks"]),
         "justification": f"Score {float(ex['gained_marks']):g} based on the rubric."},
        ensure_ascii=False,
    )


def _turn(role: str, text: str, with_image: bool = False) -> dict:
    content = []
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    return {"role": role, "content": content}


def build_messages(user_text: str,
                   assistant_text: str | None = None,
                   exemplars: list[dict] | None = None,
                   system_in_user: bool = False) -> list[dict]:
    """Chat messages for any regime.

    exemplars: list of {user_text, assistant_text} — each contributes one
    image, in order, BEFORE the query image. Callers must pass images to the
    processor in that same order.

    system_in_user: fold SYSTEM_PROMPT into the first user turn, for chat
    templates that do not support a system role.
    """
    msgs = []
    if not system_in_user:
        msgs.append(_turn("system", SYSTEM_PROMPT))

    first_user_prefix = f"{SYSTEM_PROMPT}\n\n" if system_in_user else ""
    for i, e in enumerate(exemplars or []):
        prefix = first_user_prefix if i == 0 else ""
        msgs.append(_turn("user", prefix + e["user_text"], with_image=True))
        msgs.append(_turn("assistant", e["assistant_text"]))
    prefix = first_user_prefix if not (exemplars or []) else ""
    msgs.append(_turn("user", prefix + user_text, with_image=True))
    if assistant_text is not None:
        msgs.append(_turn("assistant", assistant_text))
    return msgs


# --------------------------------------------------------------------------
# Donut (encoder-decoder, OCR-free)
# --------------------------------------------------------------------------

def build_donut_prompt(ex: dict, max_chars: int = 1200) -> str:
    """Compact decoder-prefix prompt for Donut.

    Donut has no chat template and a short BART decoder (1536 positions), so
    the prompt is a flat, heavily abbreviated string. The reference answer is
    the first thing trimmed — the rubric and criterion matter more for
    grading, and the student's own words come from the image.

    Truncating TEXT here is safe: unlike the chat VLMs, Donut's image never
    becomes decoder tokens, so cutting text cannot corrupt image placeholders.
    """
    try:
        levels = json.loads(ex["rubric_levels"])
    except Exception:
        levels = {}
    levels_str = " ".join(f"{k}: {v}" for k, v in levels.items())
    ref = (ex.get("model_answer") or "").strip().replace("\n", " ")
    budget = max(0, max_chars - len(levels_str) - len(str(ex["criterion_name"])))
    if len(ref) > budget:
        ref = ref[:budget] + "…"
    mx = criterion_max_of(ex)
    return (f"<criterion>{ex['criterion_name']} (max {mx:g})"
            f"<rubric>{levels_str}"
            f"<reference>{ref}"
            f"<score>")


def donut_target(ex: dict) -> str:
    """Gold decoder target for Donut — a bare number keeps the short decoder
    focused on the quantity being predicted."""
    return f"{float(ex['gained_marks']):g}"


# Cache of processors known to reject the system role, so the fallback is
# only discovered once per process.
_NO_SYSTEM: set[int] = set()


def apply_chat_template_safe(processor, *, user_text: str,
                             assistant_text: str | None = None,
                             exemplars: list[dict] | None = None,
                             add_generation_prompt: bool = False) -> str:
    """apply_chat_template with an automatic no-system-role fallback."""
    key = id(processor)
    for system_in_user in ((True,) if key in _NO_SYSTEM else (False, True)):
        msgs = build_messages(user_text, assistant_text, exemplars,
                              system_in_user=system_in_user)
        try:
            return processor.apply_chat_template(
                msgs, tokenize=False,
                add_generation_prompt=add_generation_prompt)
        except Exception:
            if not system_in_user:
                _NO_SYSTEM.add(key)
                continue
            raise
    raise RuntimeError("chat template failed")  # pragma: no cover
