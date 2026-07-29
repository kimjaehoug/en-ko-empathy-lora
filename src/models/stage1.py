from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_lora_lm(
    model_name: str,
    *,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if target_modules is None:
        # gpt2-style
        target_modules = ["c_attn", "c_proj"]
        # fallback for llama/qwen-like if those modules absent
        module_names = {n.split(".")[-1] for n, _ in model.named_modules()}
        if "q_proj" in module_names:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, cfg)
    return model


class Stage1EmpathyModel(nn.Module):
    """EN Stage1: LoRA causal LM + affect classification head.

    Factor LoRA bank is represented as a named PEFT adapter (affect). Additional
    adapters (strategy/relation/culture) can be added in later stages without
    changing the backbone.
    """

    def __init__(
        self,
        lm: nn.Module,
        n_emotions: int,
        *,
        emotion_loss_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.lm = lm
        hidden = lm.config.hidden_size
        self.emotion_head = nn.Linear(hidden, n_emotions)
        self.emotion_loss_weight = emotion_loss_weight
        self.n_emotions = n_emotions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        emotion_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        lm_loss = outputs.loss

        hidden = outputs.hidden_states[-1]
        # Pool prompt tokens: positions supervised as -100 and not padding
        prompt_mask = (labels == -100) & (attention_mask == 1)
        pooled = self._masked_mean(hidden, prompt_mask)
        logits = self.emotion_head(pooled)

        loss = lm_loss
        emotion_loss = None
        if emotion_ids is not None:
            valid = emotion_ids >= 0
            if valid.any():
                emotion_loss = nn.functional.cross_entropy(logits[valid], emotion_ids[valid])
                loss = lm_loss + self.emotion_loss_weight * emotion_loss

        return {
            "loss": loss,
            "lm_loss": lm_loss.detach(),
            "emotion_loss": None if emotion_loss is None else emotion_loss.detach(),
            "emotion_logits": logits,
            "logits": outputs.logits,
        }

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom
