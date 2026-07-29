from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

from .stage1 import build_tokenizer


class Stage3EmpathyModel(nn.Module):
    """KO Stage3: gated LoRA LM + affect/strategy/relation heads."""

    def __init__(
        self,
        lm: nn.Module,
        *,
        n_emotions: int,
        n_strategies: int,
        n_relations: int,
        emotion_loss_weight: float = 0.2,
        strategy_loss_weight: float = 0.3,
        relation_loss_weight: float = 0.2,
        compose_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.lm = lm
        hidden = lm.config.hidden_size
        self.emotion_head = nn.Linear(hidden, n_emotions)
        self.strategy_head = nn.Linear(hidden, n_strategies)
        self.relation_head = nn.Linear(hidden, n_relations)
        self.emotion_loss_weight = emotion_loss_weight
        self.strategy_loss_weight = strategy_loss_weight
        self.relation_loss_weight = relation_loss_weight
        # composer stub: scale residual adapters uniformly for now
        self.compose_alpha = compose_alpha

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        emotion_ids: torch.Tensor | None = None,
        strategy_ids: torch.Tensor | None = None,
        relation_ids: torch.Tensor | None = None,
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

        prompt_mask = (labels == -100) & (attention_mask == 1)
        pooled = self._masked_mean(hidden, prompt_mask)

        emotion_logits = self.emotion_head(pooled)
        strategy_logits = self.strategy_head(pooled)
        relation_logits = self.relation_head(pooled)

        loss = lm_loss
        losses = {"lm_loss": lm_loss.detach()}

        def add_ce(logits, ids, weight, name):
            nonlocal loss
            if ids is None:
                losses[name] = None
                return
            valid = ids >= 0
            if valid.any():
                ce = nn.functional.cross_entropy(logits[valid], ids[valid])
                loss = loss + weight * ce
                losses[name] = ce.detach()
            else:
                losses[name] = None

        add_ce(emotion_logits, emotion_ids, self.emotion_loss_weight, "emotion_loss")
        add_ce(strategy_logits, strategy_ids, self.strategy_loss_weight, "strategy_loss")
        add_ce(relation_logits, relation_ids, self.relation_loss_weight, "relation_loss")

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


def build_stage3_lm(
    *,
    model_name: str,
    stage1_lora_dir: str,
    gates: dict,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    force_init: str | None = None,
):
    """Create KO LoRA from EN adapter with share/relearn/suppress policy.

    Prototype policy on a single adapter:
    - if majority share -> load EN LoRA weights
    - if any suppress/relearn dominant on strategy/relation/culture -> re-init fresh LoRA

    force_init: optional override in
      {"share", "relearn", "suppress", "affect_priority", "auto"/None}.
    """
    base = AutoModelForCausalLM.from_pretrained(model_name)
    decisions = [v.get("decision") for v in gates.values()]
    share_n = sum(d == "share" for d in decisions)
    relearn_like = sum(d in {"relearn", "suppress"} for d in decisions)
    affect_decision = (gates.get("affect") or {}).get("decision", "relearn")

    module_names = {n.split(".")[-1] for n, _ in base.named_modules()}
    if "q_proj" in module_names:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    else:
        target_modules = ["c_attn", "c_proj"]

    mode = (force_init or "auto").lower()
    if mode == "affect_priority":
        use_share = affect_decision == "share"
    elif mode == "suppress":
        use_share = True
    else:
        use_share = mode == "share" or (mode == "auto" and share_n >= relearn_like)

    if use_share:
        # keep EN knowledge
        lm = PeftModel.from_pretrained(base, stage1_lora_dir)
        lm.train()
        for n, p in lm.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        init_mode = "share_from_en"
    else:
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        lm = get_peft_model(base, cfg)
        init_mode = "relearn_fresh"

    # suppress: scale down LoRA params slightly as soft unlearn of EN directions
    do_suppress = mode == "suppress" or (
        mode == "auto" and "suppress" in decisions and init_mode == "share_from_en"
    )
    if do_suppress and init_mode == "share_from_en":
        with torch.no_grad():
            for n, p in lm.named_parameters():
                if "lora_" in n:
                    p.mul_(0.5)
        init_mode = "share_then_suppress_scale"

    if mode == "affect_priority":
        init_mode = f"affect_priority_{init_mode}"

    return lm, init_mode
