"""Single-source configuration loading.

All hyperparameters live in config.yaml at the repo root. Every script calls
`load_config()` (optionally with a --config override) and passes the resulting
dict-like Config object around.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

VALID_REGIMES = ("zero_shot", "few_shot", "lora", "full")

# regimes that train weights (and therefore write a checkpoint)
TRAINING_REGIMES = ("lora", "full")


class Config(dict):
    """dict with attribute access and nested-key helper: cfg.get_in('a.b.c')."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def get(self, k: str, default: Any = None) -> Any:
        v = super().get(k, default)
        return Config(v) if isinstance(v, dict) else v

    def get_in(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node


def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml. Path resolution order:
    explicit arg -> $OBE_CONFIG -> <repo>/config.yaml
    """
    path = Path(path) if path else Path(
        os.environ.get("OBE_CONFIG", DEFAULT_CONFIG_PATH))
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    cfg["_config_path"] = str(path)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    for key in ("paths", "dataset", "models", "regimes",
                "training", "inference", "ensemble"):
        if key not in cfg:
            raise ValueError(f"config.yaml missing required section: {key}")
    for r in cfg["regimes"]:
        if r not in VALID_REGIMES:
            raise ValueError(f"unknown regime '{r}' (valid: {VALID_REGIMES})")
    for name, m in cfg["models"].items():
        if "model_id" not in m:
            raise ValueError(f"models.{name} missing model_id")
    ens = cfg["ensemble"]
    known = set(cfg["models"]) | {"ensemble"}
    for m in list(ens["small_models"]) + [ens["tiebreaker"]]:
        if m not in known:
            raise ValueError(f"ensemble references unknown model '{m}'")
    if cfg["dataset"].get("source") not in ("hf_hub", "local"):
        raise ValueError("dataset.source must be 'hf_hub' or 'local'")
    for key in ("score_resolution", "max_criterion_score"):
        value = float(cfg["dataset"].get(key, 0))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"dataset.{key} must be a positive finite number")
    ens_resolution = float(cfg["ensemble"].get("score_resolution", 0))
    if not np.isfinite(ens_resolution) or ens_resolution <= 0:
        raise ValueError("ensemble.score_resolution must be positive and finite")
    if int(cfg.get("few_shot", {}).get("n_exemplars", 0)) < 0:
        raise ValueError("few_shot.n_exemplars must be non-negative")
    fraction = float(cfg.get("training", {}).get("full", {}).get(
        "chat_unfreeze_fraction", 0.30))
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError(
            "training.full.chat_unfreeze_fraction must be in (0, 1]")
    for name, model_cfg in cfg["models"].items():
        for key in ("batch_size_train", "grad_accum", "batch_size_full",
                    "grad_accum_full", "batch_size_infer"):
            if key in model_cfg and int(model_cfg[key]) < 1:
                raise ValueError(f"models.{name}.{key} must be >= 1")


def save_config(cfg: Config, path: str | Path) -> Path:
    """Persist a (possibly modified) config — used by the Colab autotuner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump = {k: v for k, v in cfg.items() if not k.startswith("_")}
    path.write_text(yaml.safe_dump(dump, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Canonical output locations (keeps every script consistent)
# --------------------------------------------------------------------------

def output_root(cfg: Config) -> Path:
    p = Path(cfg.paths.output_root)
    return p if p.is_absolute() else REPO_ROOT / p


def run_dir(cfg: Config, model: str, regime: str) -> Path:
    return output_root(cfg) / "runs" / model / regime


def predictions_path(cfg: Config, model: str, regime: str, run: int = 0) -> Path:
    """Main (deterministic) prediction file used by Studies 1, 2, 4, 5."""
    return run_dir(cfg, model, regime) / f"predictions_run{run}.json"


def reliability_dir(cfg: Config, model: str, regime: str) -> Path:
    """Study 3 repeated stochastic runs live in their own folder so they can
    never overwrite the deterministic run0 the other studies rely on."""
    return run_dir(cfg, model, regime) / "reliability"


def reliability_path(cfg: Config, model: str, regime: str, run: int) -> Path:
    return reliability_dir(cfg, model, regime) / f"predictions_run{run}.json"


def study_dir(cfg: Config, study: int) -> Path:
    return output_root(cfg) / "studies" / f"study{study}"


def checkpoint_dir(cfg: Config, model: str, regime: str = "lora") -> Path:
    """Trained weights for one (model, regime).

    The regime is part of the path: a LoRA adapter and a resource-aware
    `full` checkpoint of the same model are different artifacts and must not
    overwrite one another.
    """
    return output_root(cfg) / "checkpoints" / model / regime


def _checkpoint_has_weights(path: Path, regime: str) -> bool:
    """True only for a loadable checkpoint, not a merely-created folder."""
    if regime == "lora":
        return ((path / "adapter_config.json").is_file() and
                ((path / "adapter_model.safetensors").is_file() or
                 (path / "adapter_model.bin").is_file()))
    return ((path / "config.json").is_file() and
            (any(path.glob("model*.safetensors")) or
             any(path.glob("pytorch_model*.bin"))))


def checkpoint_artifact_dir(cfg: Config, model: str,
                            regime: str = "lora") -> Path | None:
    """Resolve a complete root checkpoint or the latest Trainer checkpoint.

    Trainer creates the output directory before doing any work and writes
    epoch checkpoints below ``checkpoint-<step>``. Directory existence alone
    therefore cannot prove that an adapter/model can be loaded.
    """
    root = checkpoint_dir(cfg, model, regime)
    if _checkpoint_has_weights(root, regime):
        return root
    if not root.is_dir():
        return None

    def step(path: Path) -> int:
        try:
            return int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    for candidate in sorted(root.glob("checkpoint-*"), key=step,
                            reverse=True):
        if candidate.is_dir() and _checkpoint_has_weights(candidate, regime):
            return candidate
    return None


def checkpoint_is_ready(cfg: Config, model: str,
                        regime: str = "lora") -> bool:
    return checkpoint_artifact_dir(cfg, model, regime) is not None


def data_path(cfg: Config, key: str) -> Path:
    """Resolve paths.<key> relative to the repo root when not absolute."""
    p = Path(cfg.paths[key])
    return p if p.is_absolute() else REPO_ROOT / p
