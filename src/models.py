"""Model registry: load any model from a config entry.

Two architecture families, selected by `models.<name>.family`:

  chat_vlm  Qwen2.5-VL, InternVL3, Pixtral — decoder-only multimodal chat
            models. Image tokens are interleaved into a chat template; the
            prompt is a conversation.

  donut     Donut (naver-clova-ix/donut-base) — an OCR-free encoder-decoder
            (Swin vision encoder + BART text decoder). It has NO chat
            template and no in-context-learning ability: the image goes to
            the encoder as pixel values, and the text prompt is a decoder
            prefix. Included to test whether an OCR-free document model can
            grade handwriting without a language-model backbone.

Supports 4/8-bit quantization (bitsandbytes), LoRA adapters, partially trained
chat-VLM checkpoints, and fully trained Donut checkpoints.
"""

from __future__ import annotations

from .config import Config

CHAT_VLM = "chat_vlm"
DONUT = "donut"


def model_float_dtype(model):
    """Return the floating parameter dtype used for model inputs."""
    import torch

    declared = getattr(model, "dtype", None)
    try:
        # PEFT can keep its small trainable adapters in FP32 while the much
        # larger base model (including image convolutions) is BF16/FP16. The
        # wrapper's reported/first dtype is therefore not necessarily the
        # correct input dtype. Choose the dtype representing the most parameter
        # elements.
        counts = {}
        for parameter in model.parameters():
            if parameter.is_floating_point():
                counts[parameter.dtype] = (
                    counts.get(parameter.dtype, 0) + parameter.numel())
        if counts:
            return max(counts, key=counts.get)
    except AttributeError:
        pass
    if isinstance(declared, torch.dtype) and declared.is_floating_point:
        return declared
    return None


def move_inputs_to_model(inputs, model) -> dict:
    """Move processor outputs to the model's input device and float dtype.

    Image processors normally emit FP32 ``pixel_values`` even when a model is
    loaded in BF16/FP16.  CUDA convolutions require their floating input and
    weight dtypes to match outside an autocast context, which is the case for
    generation and XAI. Integer tensors such as token IDs, masks, grids, and
    image sizes retain their original dtype.
    """
    import torch

    device = getattr(model, "device", None)
    dtype = model_float_dtype(model)
    if device is None or getattr(device, "type", None) == "meta":
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = None
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if value.is_floating_point() and dtype is not None:
            kwargs["dtype"] = dtype
        moved[key] = value.to(non_blocking=(device is not None and
                                             getattr(device, "type", None)
                                             == "cuda"),
                              **kwargs)
    return moved


def family_of(cfg: Config, model_key: str) -> str:
    return str(Config(cfg.models[model_key]).get("family", CHAT_VLM))


def is_donut(cfg: Config, model_key: str) -> bool:
    return family_of(cfg, model_key) == DONUT


def _dtype_kwarg() -> str:
    """transformers >= 4.56 renamed `torch_dtype` to `dtype`.

    `from_pretrained` swallows unknown keys via **kwargs, so passing the wrong
    name is silently ignored and the model loads in float32 (and OOMs). Decide
    from the installed version instead of guessing.
    """
    try:
        import transformers
        parts = str(transformers.__version__).split(".")
        major, minor = int(parts[0]), int(parts[1])
        return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    except Exception:
        return "torch_dtype"


def _quant_config(lcfg: Config):
    if not (lcfg.get("load_in_4bit") or lcfg.get("load_in_8bit")):
        return None
    import torch
    from transformers import BitsAndBytesConfig

    if lcfg.get("load_in_4bit"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, lcfg.torch_dtype),
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_model_and_processor(cfg: Config, model_key: str,
                             adapter_dir: str | None = None,
                             full_checkpoint: str | None = None,
                             quantize: bool | None = None):
    """Instantiate processor + model for `model_key`.

    adapter_dir     : attach a trained LoRA adapter (PEFT) on top.
    full_checkpoint : load a resource-aware `full` checkpoint instead of the
                      base model (partial chat VLM or fully trained Donut).
    quantize        : override model_loading.load_in_4bit/8bit (full
                      fine-tuning forces this off — you cannot train
                      quantized base weights end to end).
    """
    import torch
    import transformers
    from transformers import AutoProcessor

    mcfg = Config(cfg.models[model_key])
    lcfg = Config(dict(cfg.model_loading))
    if quantize is False:
        lcfg["load_in_4bit"] = False
        lcfg["load_in_8bit"] = False

    family = str(mcfg.get("family", CHAT_VLM))
    source = full_checkpoint or mcfg.model_id
    dtype = getattr(torch, lcfg.torch_dtype)

    kwargs = dict(device_map=lcfg.device_map,
                  trust_remote_code=lcfg.trust_remote_code)
    kwargs[_dtype_kwarg()] = dtype
    qc = _quant_config(lcfg)
    if qc is not None:
        kwargs["quantization_config"] = qc
        print(f"    quantization: "
              f"{'4-bit nf4' if lcfg.get('load_in_4bit') else '8-bit'}")

    print(f"=== Loading {source} "
          f"({mcfg.model_class}, family={family}) ===")

    if family == DONUT:
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        processor = DonutProcessor.from_pretrained(
            full_checkpoint or mcfg.model_id)
        # Donut is ~200M params: no device_map sharding, no quantization.
        model = VisionEncoderDecoderModel.from_pretrained(
            source, **{_dtype_kwarg(): dtype})
        if torch.cuda.is_available():
            model = model.to("cuda")
        _configure_donut(cfg, model, processor)
    else:
        cls = getattr(transformers, mcfg.model_class, None)
        if cls is None:  # graceful fallback across transformers versions
            from transformers import AutoModelForImageTextToText as cls  # noqa
            print(f"    [warn] {mcfg.model_class} not in transformers "
                  f"{transformers.__version__}; using "
                  f"AutoModelForImageTextToText")
        # eager attention is REQUIRED so output_attentions=True returns real
        # tensors during the XAI step (SDPA / FlashAttention return None).
        kwargs["attn_implementation"] = lcfg.attn_implementation
        processor = AutoProcessor.from_pretrained(
            mcfg.model_id, trust_remote_code=lcfg.trust_remote_code)
        try:
            model = cls.from_pretrained(source, **kwargs)
        except TypeError:
            kwargs.pop("attn_implementation", None)
            model = cls.from_pretrained(source, **kwargs)
            print("    [warn] loaded without attn_implementation='eager'; "
                  "XAI attention rollout will be unavailable")
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

    if adapter_dir:
        from peft import PeftModel
        print(f"    attaching LoRA adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)

    return processor, model


def _configure_donut(cfg: Config, model, processor) -> None:
    """Set the decoder start / pad / eos ids and the image size Donut needs."""
    dcfg = Config(cfg.get("donut", {}) or {})
    tok = processor.tokenizer
    task_token = str(dcfg.get("task_start_token", "<s_grade>"))

    added = tok.add_special_tokens({"additional_special_tokens": [task_token]})
    if added:
        model.decoder.resize_token_embeddings(len(tok))

    model.config.decoder_start_token_id = tok.convert_tokens_to_ids(task_token)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    if getattr(model.config, "decoder", None) is not None:
        model.config.decoder.is_decoder = True
        model.config.decoder.add_cross_attention = True

    side = dcfg.get("image_size")
    if side:
        # Donut expects [height, width]; tall answer strips keep their aspect
        processor.image_processor.size = {"height": int(side[0]),
                                          "width": int(side[1])}
    print(f"    donut: task token {task_token!r} "
          f"(id {model.config.decoder_start_token_id}), "
          f"image size {processor.image_processor.size}")


def set_padding_side(processor, side: str) -> None:
    """Request a padding side on the processor and its tokenizer.

    This is a *request*, not a guarantee. A multimodal processor merges its
    own class-level `_defaults["text_kwargs"]` over the tokenizer attributes,
    so a processor that declares a padding side wins over the value written
    here. Call sites that depend on the side must pass it explicitly (see
    `processor_accepts`) and/or read the returned attention mask.
    """
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = side
    if hasattr(processor, "padding_side"):
        try:
            processor.padding_side = side
        except Exception:
            pass


def processor_accepts(processor, name: str) -> bool:
    """True when ``processor.__call__`` accepts the keyword ``name``.

    Transformers processors take ``**kwargs`` typed with their per-modality
    ``ProcessingKwargs``, so a variadic keyword parameter means the flat
    keyword is routed and honored. Minimal/stub processors with a fixed
    signature return False, and the caller must not pass the keyword.
    """
    import inspect

    call = getattr(processor, "__call__", None)
    if call is None:
        return False
    try:
        parameters = inspect.signature(call).parameters
    except (TypeError, ValueError):
        return False
    if name in parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD
               for parameter in parameters.values())


def gpu_memory_gb() -> float:
    """Total VRAM of device 0 in GiB (0.0 when no CUDA device)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        return 0.0


def count_parameters(model) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def estimate_full_finetune_gb(model) -> float:
    """Rough VRAM needed for full fine-tuning with AdamW in mixed precision:
    weights(2) + grads(2) + Adam m/v(8) bytes per parameter, plus activations.
    """
    _, total = count_parameters(model)
    return total * 12 / 1024 ** 3


def estimate_selected_finetune_gb(model) -> float:
    """Conservative primary-GPU estimate for the current trainable subset.

    All parameters consume their loaded storage. Trainable parameters also
    need a gradient plus two FP32 AdamW moment tensors. This matches the
    project's historical 12-bytes/parameter full-tuning estimate for a BF16
    model while correctly scaling optimizer/gradient memory for partial FT.
    Activations are deliberately left for the safety margin in the caller.
    """
    weight_bytes = sum(p.numel() * p.element_size()
                       for p in model.parameters())
    training_bytes = sum(
        p.numel() * (p.element_size() + 8)
        for p in model.parameters() if p.requires_grad)
    return (weight_bytes + training_bytes) / 1024 ** 3
