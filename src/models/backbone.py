"""Shared backbone load helpers for SELENE (bf16 CUDA / PEFT LoRA)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN_ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
QWEN_FULL_TARGETS = QWEN_ATTN_TARGETS + ["gate_proj", "up_proj", "down_proj"]
GPT2_TARGETS = ["c_attn", "c_proj"]


def resolve_torch_dtype(dtype: str | torch.dtype | None = "bf16") -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    name = (dtype or "bf16").lower()
    if name in {"bf16", "bfloat16"}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name == "auto":
        return resolve_torch_dtype("bf16")
    raise ValueError(f"unsupported dtype: {dtype}")


def infer_lora_targets(
    model: torch.nn.Module,
    *,
    target_modules: list[str] | None = None,
    include_mlp: bool = False,
) -> list[str]:
    if target_modules:
        return list(target_modules)
    module_names = {n.split(".")[-1] for n, _ in model.named_modules()}
    if "q_proj" in module_names:
        return list(QWEN_FULL_TARGETS if include_mlp else QWEN_ATTN_TARGETS)
    return list(GPT2_TARGETS)


def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Causal LM training: pad on the right for labels alignment
    tok.padding_side = "right"
    return tok


def load_base_causal_lm(
    model_name: str,
    *,
    dtype: str | torch.dtype | None = "bf16",
    device_map: str | dict | None = None,
    trust_remote_code: bool = True,
):
    """Load frozen-backbone-ready Causal LM in bf16 (no 4-bit / QLoRA)."""
    torch_dtype = resolve_torch_dtype(dtype)
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    # transformers>=4.56 prefers `dtype=`; older builds used `torch_dtype=`
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, **kwargs
        )
    model.config.use_cache = False
    return model


def cuda_mem_gb() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"allocated_gb": None, "reserved_gb": None, "peak_allocated_gb": None}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }


def git_commit_hash(repo_root: str | Path | None = None) -> str | None:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None
