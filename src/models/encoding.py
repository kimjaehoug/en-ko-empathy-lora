from __future__ import annotations

import torch
import torch.nn as nn
from peft import PeftModel

from .backbone import build_tokenizer, load_base_causal_lm
from .stage1 import Stage1EmpathyModel


def pick_device(name: str = "auto") -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        return torch.device("mps")
    if name == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_stage1_bundle(
    *,
    model_name: str,
    stage1_dir: str,
    n_emotions: int,
    device: torch.device,
    dtype: str | torch.dtype | None = "bf16",
) -> tuple[Stage1EmpathyModel, any]:
    stage1_dir = str(stage1_dir)
    tok = build_tokenizer(model_name)
    base = load_base_causal_lm(model_name, dtype=dtype)
    lm = PeftModel.from_pretrained(base, f"{stage1_dir}/lora")
    model = Stage1EmpathyModel(lm, n_emotions=n_emotions)
    head_path = f"{stage1_dir}/emotion_head.pt"
    state = torch.load(head_path, map_location="cpu")
    model.emotion_head.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def encode_prompts(
    model: Stage1EmpathyModel,
    tokenizer,
    prompts: list[str],
    *,
    device: torch.device,
    max_length: int = 256,
    batch_size: int = 4,
    progress_cb=None,
) -> torch.Tensor:
    """Return pooled prompt embeddings [N, H]."""
    vecs = []
    model.eval()
    total = len(prompts)
    for i in range(0, total, batch_size):
        chunk = prompts[i : i + batch_size]
        tok = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        out = model.lm(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]
        mask = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        vecs.append(pooled.detach().float().cpu())
        if progress_cb is not None:
            progress_cb(min(i + batch_size, total), total)
    return torch.cat(vecs, dim=0)


def cosine_mean_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    x = nn.functional.normalize(x.mean(dim=0), dim=0)
    y = nn.functional.normalize(y.mean(dim=0), dim=0)
    return float((x * y).sum().item())


def fit_linear_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_classes: int,
    lr: float = 1e-1,
    steps: int = 200,
) -> tuple[nn.Linear, float]:
    """Tiny CPU probe; returns model and train accuracy."""
    device = torch.device("cpu")
    x = x.to(device).float()
    y = y.to(device)
    valid = y >= 0
    x, y = x[valid], y[valid]
    if len(y) == 0 or n_classes <= 1:
        raise ValueError("not enough labeled samples for probe")

    probe = nn.Linear(x.size(1), n_classes)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = probe(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(x).argmax(dim=-1)
        acc = float((pred == y).float().mean().item())
    return probe, acc


@torch.no_grad()
def probe_accuracy(probe: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> float:
    valid = y >= 0
    if valid.sum() == 0:
        return float("nan")
    pred = probe(x[valid].float()).argmax(dim=-1)
    return float((pred == y[valid]).float().mean().item())
