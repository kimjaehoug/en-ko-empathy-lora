from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from .backbone import build_tokenizer, infer_lora_targets, load_base_causal_lm

__all__ = [
    "Stage3EmpathyModel",
    "build_stage3_lm",
    "build_tokenizer",
    "snapshot_lora_params",
    "lora_anchor_loss",
]


def _mlp_head(hidden: int, n_out: int, *, dropout: float = 0.1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_out),
    )


class Stage3EmpathyModel(nn.Module):
    """KO Stage3: gated LoRA LM + A/S/R heads.

    Strategy uses multi-label BCE when ``strategy_multihot`` is provided
    (AI Hub S_strategy is a set of empathy strategies, not a single class).

    When ``two_pass_affect`` is True, emotion logits are computed from a
    second forward with LoRA adapters disabled (EN-merged / share path),
    while LM/S/R use the active KO relearn LoRA path.
    """

    def __init__(
        self,
        lm: nn.Module,
        *,
        n_emotions: int,
        n_strategies: int,
        n_relations: int,
        emotion_loss_weight: float = 0.2,
        strategy_loss_weight: float = 1.0,
        relation_loss_weight: float = 0.2,
        compose_alpha: float = 1.0,
        strategy_multilabel: bool = True,
        deep_strategy_head: bool = True,
        two_pass_affect: bool = False,
        emotion_class_weights: torch.Tensor | None = None,
        strategy_pos_weight: torch.Tensor | None = None,
        relation_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lm = lm
        hidden = lm.config.hidden_size
        self.emotion_head = nn.Linear(hidden, n_emotions)
        if deep_strategy_head:
            self.strategy_head = _mlp_head(hidden, n_strategies)
        else:
            self.strategy_head = nn.Linear(hidden, n_strategies)
        self.relation_head = nn.Linear(hidden, n_relations)
        self.emotion_loss_weight = emotion_loss_weight
        self.strategy_loss_weight = strategy_loss_weight
        self.relation_loss_weight = relation_loss_weight
        self.compose_alpha = compose_alpha
        self.strategy_multilabel = strategy_multilabel
        self.two_pass_affect = two_pass_affect
        self.emotion_class_weights = emotion_class_weights
        self.strategy_pos_weight = strategy_pos_weight
        self.relation_class_weights = relation_class_weights

    def _pool(self, hidden: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        prompt_mask = (labels == -100) & (attention_mask == 1)
        return self._masked_mean(hidden, prompt_mask).float()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        emotion_ids: torch.Tensor | None = None,
        strategy_ids: torch.Tensor | None = None,
        relation_ids: torch.Tensor | None = None,
        strategy_multihot: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        if outputs.loss is None or torch.isnan(outputs.loss):
            lm_loss = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        else:
            lm_loss = outputs.loss

        pooled = self._pool(hidden, labels, attention_mask)

        # Factor-bank: A from share path (adapters off = EN-merged base only)
        if self.two_pass_affect and hasattr(self.lm, "disable_adapter"):
            with self.lm.disable_adapter():
                out_a = self.lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            pooled_a = self._pool(out_a.hidden_states[-1], labels, attention_mask)
            emotion_logits = self.emotion_head(pooled_a)
        else:
            emotion_logits = self.emotion_head(pooled)

        strategy_logits = self.strategy_head(pooled)
        relation_logits = self.relation_head(pooled)

        loss = lm_loss
        losses: dict = {"lm_loss": lm_loss.detach()}

        # ---- Affect (single-label CE) ----
        if emotion_ids is not None:
            valid = emotion_ids >= 0
            if valid.any():
                w = self.emotion_class_weights
                if w is not None:
                    w = w.to(emotion_logits.device)
                ce = nn.functional.cross_entropy(
                    emotion_logits[valid], emotion_ids[valid], weight=w
                )
                loss = loss + self.emotion_loss_weight * ce
                losses["emotion_loss"] = ce.detach()
            else:
                losses["emotion_loss"] = None
        else:
            losses["emotion_loss"] = None

        # ---- Strategy (multi-label BCE preferred) ----
        if self.strategy_multilabel and strategy_multihot is not None:
            target = strategy_multihot.float()
            row_has = target.sum(dim=-1) > 0
            if row_has.any():
                pw = self.strategy_pos_weight
                if pw is not None:
                    pw = pw.to(strategy_logits.device)
                bce = nn.functional.binary_cross_entropy_with_logits(
                    strategy_logits[row_has],
                    target[row_has],
                    pos_weight=pw,
                )
                loss = loss + self.strategy_loss_weight * bce
                losses["strategy_loss"] = bce.detach()
            else:
                losses["strategy_loss"] = None
        elif strategy_ids is not None:
            valid = strategy_ids >= 0
            if valid.any():
                ce = nn.functional.cross_entropy(
                    strategy_logits[valid], strategy_ids[valid]
                )
                loss = loss + self.strategy_loss_weight * ce
                losses["strategy_loss"] = ce.detach()
            else:
                losses["strategy_loss"] = None
        else:
            losses["strategy_loss"] = None

        # ---- Relation (single-label CE) ----
        if relation_ids is not None:
            valid = relation_ids >= 0
            if valid.any():
                w = self.relation_class_weights
                if w is not None:
                    w = w.to(relation_logits.device)
                ce = nn.functional.cross_entropy(
                    relation_logits[valid], relation_ids[valid], weight=w
                )
                loss = loss + self.relation_loss_weight * ce
                losses["relation_loss"] = ce.detach()
            else:
                losses["relation_loss"] = None
        else:
            losses["relation_loss"] = None

        return {
            "loss": loss,
            **losses,
            "emotion_logits": emotion_logits,
            "strategy_logits": strategy_logits,
            "relation_logits": relation_logits,
        }

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom


def snapshot_lora_params(lm: nn.Module) -> dict[str, torch.Tensor]:
    """CPU float32 copy of LoRA weights for anchor regularization."""
    snap = {}
    for n, p in lm.named_parameters():
        if "lora_" in n and p.requires_grad:
            snap[n] = p.detach().float().cpu().clone()
    return snap


def lora_anchor_loss(lm: nn.Module, snapshot: dict[str, torch.Tensor]) -> torch.Tensor:
    if not snapshot:
        device = next(lm.parameters()).device
        return torch.zeros((), device=device)
    total = None
    n = 0
    for name, p in lm.named_parameters():
        if name not in snapshot:
            continue
        diff = (p.float() - snapshot[name].to(p.device)).pow(2).mean()
        total = diff if total is None else total + diff
        n += 1
    if total is None:
        device = next(lm.parameters()).device
        return torch.zeros((), device=device)
    return total / max(n, 1)


def build_stage3_lm(
    *,
    model_name: str,
    stage1_lora_dir: str,
    gates: dict,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    force_init: str | None = None,
    dtype: str | torch.dtype | None = "bf16",
    include_mlp: bool = False,
    target_modules: list[str] | None = None,
    device_map: str | dict | None = None,
):
    """Create KO LoRA from EN adapter with share/relearn/suppress/soft_share policy.

    force_init:
      share | relearn | suppress | affect_priority | soft_share | select | select_dual | select_bank | auto
    soft_share: always init from EN LoRA (Dir II retention), gates only scale LR/suppress.
    select: EN LoRA init, never suppress-scale; intended with freeze_shared_lora / head-only adapt.
    select_dual / select_bank: merge EN LoRA into base (share Affect), then train a fresh KO LoRA
      (relearn S/R + LM). select_bank is the Factor-LoRA Bank entrypoint (same init; training
      enables two_pass_affect / stronger S·R weights in the trainer).
    """
    base = load_base_causal_lm(model_name, dtype=dtype, device_map=device_map)
    decisions = [v.get("decision") for v in gates.values()]
    share_n = sum(d == "share" for d in decisions)
    relearn_like = sum(d in {"relearn", "suppress"} for d in decisions)
    affect_decision = (gates.get("affect") or {}).get("decision", "relearn")

    targets = infer_lora_targets(
        base, target_modules=target_modules, include_mlp=include_mlp
    )

    mode = (force_init or "auto").lower()
    if mode == "madx":
        # MAD-X-style: frozen EN language adapter + trainable task adapter (Pfeiffer et al. 2020).
        lm = PeftModel.from_pretrained(base, stage1_lora_dir, adapter_name="language")
        for n, p in lm.named_parameters():
            if "language" in n and "lora_" in n:
                p.requires_grad = False
        task_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=targets,
            bias="none",
        )
        lm.add_adapter("task", task_cfg)
        lm.set_adapter(["language", "task"])
        lm.train()
        for n, p in lm.named_parameters():
            if "task" in n and "lora_" in n:
                p.requires_grad = True
        return lm, f"madx_lang_frozen_task_r{lora_r}"

    if mode in {"select_dual", "select_bank"}:
        # Bake EN adapter into weights, then attach trainable KO LoRA (factor-modular).
        en_wrapped = PeftModel.from_pretrained(base, stage1_lora_dir)
        merged = en_wrapped.merge_and_unload()
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=targets,
            bias="none",
        )
        lm = get_peft_model(merged, cfg)
        lm.train()
        tag = "select_bank_en_merged" if mode == "select_bank" else "select_dual_en_merged"
        return lm, f"{tag}_r{lora_r}"

    if mode == "affect_priority":
        use_share = affect_decision == "share"
    elif mode in {"soft_share", "select"}:
        # Paper SELECT: keep EN subspace; adapt via heads / light LoRA / gate losses
        use_share = True
    elif mode == "suppress":
        use_share = True
    else:
        use_share = mode == "share" or (mode == "auto" and share_n >= relearn_like)

    if use_share:
        lm = PeftModel.from_pretrained(base, stage1_lora_dir)
        lm.train()
        for n, p in lm.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        if mode == "select":
            init_mode = "select_from_en"
        elif mode == "soft_share":
            init_mode = "soft_share_from_en"
        else:
            init_mode = "share_from_en"
    else:
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=targets,
            bias="none",
        )
        lm = get_peft_model(base, cfg)
        init_mode = "relearn_fresh"

    # SELECT never shrinks EN LoRA at init (Dir II protect). Blind/suppress may.
    do_suppress = mode == "suppress" or (
        mode in {"auto", "soft_share"}
        and "suppress" in decisions
        and "share" in init_mode
    )
    if do_suppress and ("share" in init_mode or init_mode.startswith("soft_share")):
        # soft_share: milder suppress so EN emotion is not wiped
        scale = 0.75 if mode == "soft_share" else 0.5
        with torch.no_grad():
            for n, p in lm.named_parameters():
                if "lora_" in n:
                    p.mul_(scale)
        init_mode = f"{init_mode}_suppress_{scale}"

    if mode == "affect_priority":
        init_mode = f"affect_priority_{init_mode}"

    return lm, init_mode
