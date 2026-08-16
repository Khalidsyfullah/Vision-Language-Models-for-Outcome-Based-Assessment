"""Optional Hugging Face Hub pushes using the locally cached login."""

from __future__ import annotations

from pathlib import Path

_ADAPTER_ALLOW = (
    "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
    "README.md", "tokenizer*", "vocab*", "merges.txt",
    "special_tokens_map.json", "preprocessor_config.json",
    "processor_config.json", "chat_template.json", "added_tokens.json",
    "generation_config.json",
)
_ADAPTER_DENY = ("checkpoint-*", "optimizer.pt", "scheduler.pt",
                 "rng_state.pth", "trainer_state.json", "training_args.bin",
                 "*.tmp", "*.lock")


def require_hf_login() -> str:
    """Confirm that ``hf auth login`` has established a cached session."""
    from huggingface_hub import HfApi

    try:
        user = HfApi().whoami()["name"]
    except Exception as error:
        raise RuntimeError(
            "No Hugging Face login is available. Run `hf auth login` first."
        ) from error
    print(f"✓ Logged in to Hugging Face as: {user}")
    return user


def push_adapter(checkpoint_dir: Path, repo: str, base_model_id: str,
                 subfolder: str) -> None:
    from huggingface_hub import HfApi, create_repo

    require_hf_login()
    print(f"=== Pushing LoRA adapter to {repo}/{subfolder} ===")
    create_repo(repo, private=True, exist_ok=True)
    readme = (f"---\nlicense: apache-2.0\nbase_model: {base_model_id}\n"
              f"tags:\n  - vision-language\n  - lora\n  - exam-grading\n"
              f"  - obe\n---\n# {repo}/{subfolder}\n\n"
              f"LoRA adapter for `{base_model_id}` fine-tuned for OBE-based "
              f"exam answer grading (per-criterion).\n")
    (Path(checkpoint_dir) / "README.md").write_text(readme)
    HfApi().upload_folder(
        folder_path=str(checkpoint_dir), repo_id=repo, repo_type="model",
        path_in_repo=subfolder,
        commit_message=f"Upload LoRA adapter to /{subfolder}",
        allow_patterns=list(_ADAPTER_ALLOW),
        ignore_patterns=list(_ADAPTER_DENY))
    print(f"✓ https://huggingface.co/{repo}/tree/main/{subfolder}")


def push_results(out_dir: Path, repo: str, subfolder: str) -> None:
    from huggingface_hub import HfApi, create_repo

    require_hf_login()
    print(f"=== Pushing results to {repo}/{subfolder} (dataset) ===")
    create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    HfApi().upload_folder(
        folder_path=str(out_dir), repo_id=repo, repo_type="dataset",
        path_in_repo=subfolder,
        commit_message=f"Upload evaluation results to /{subfolder}")
    print(f"✓ https://huggingface.co/datasets/{repo}/tree/main/{subfolder}")
