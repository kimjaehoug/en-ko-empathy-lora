from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


ROLE_PREFIX = {
    "speaker": "Speaker",
    "listener": "Listener",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_dialogue_prompt(
    row: dict,
    *,
    max_history: int = 8,
    lang_hint: str | None = None,
) -> tuple[str, str]:
    """Build (prompt, target_listener_utterance) for next-listener generation."""
    situation = row.get("situation") or ""
    dialogue = row.get("dialogue") or []

    target_idx = None
    for i in range(len(dialogue) - 1, -1, -1):
        if dialogue[i].get("role") == "listener":
            target_idx = i
            break

    if target_idx is None:
        history = dialogue
        target = ""
    else:
        history = dialogue[:target_idx]
        target = dialogue[target_idx].get("text") or ""

    if max_history > 0 and len(history) > max_history:
        history = history[-max_history:]

    header = "Generate an empathetic listener response."
    if lang_hint == "ko":
        header = "한국어로 공감적인 경청자 응답을 생성하세요."

    lines = [header, f"Situation: {situation}", "Dialogue:"]
    for u in history:
        role = ROLE_PREFIX.get(u.get("role") or "speaker", "Speaker")
        text = u.get("text") or ""
        lines.append(f"{role}: {text}")
    lines.append("Listener:")
    return "\n".join(lines), target


def _infer_labels(rows: list[dict], key: str) -> list[str]:
    labels: set[str] = set()
    for row in rows:
        v = (row.get("axes") or {}).get(key)
        if isinstance(v, list):
            labels.update(str(x) for x in v if x is not None)
        elif v is not None:
            labels.add(str(v))
    return sorted(labels)


def _first_or_none(v):
    if isinstance(v, list):
        return str(v[0]) if v else None
    return None if v is None else str(v)


class EmpathyJsonlDataset(Dataset):
    """Unified processed JSONL dataset with optional multitask axis ids."""

    def __init__(
        self,
        path: str | Path,
        *,
        require_listener_target: bool = True,
        max_history: int = 8,
        emotion_labels: list[str] | None = None,
        strategy_labels: list[str] | None = None,
        relation_labels: list[str] | None = None,
        multitask: bool = False,
        lang_hint: str | None = None,
        strategy_scope: str = "utterance",
    ) -> None:
        self.path = Path(path)
        self.max_history = max_history
        self.multitask = multitask
        self.lang_hint = lang_hint
        self.strategy_scope = strategy_scope if strategy_scope in {"utterance", "session"} else "utterance"
        rows = load_jsonl(self.path)

        self.emotion_labels = emotion_labels or _infer_labels(rows, "A_affect")
        self.strategy_labels = strategy_labels or _infer_labels(rows, "S_strategy")
        self.relation_labels = relation_labels or _infer_labels(rows, "R_relation")
        self.emotion2id = {l: i for i, l in enumerate(self.emotion_labels)}
        self.strategy2id = {l: i for i, l in enumerate(self.strategy_labels)}
        self.relation2id = {l: i for i, l in enumerate(self.relation_labels)}

        self.rows: list[dict[str, Any]] = []
        for row in rows:
            prompt, target = format_dialogue_prompt(
                row, max_history=max_history, lang_hint=lang_hint
            )
            if require_listener_target and not target.strip():
                continue

            a = _first_or_none((row.get("axes") or {}).get("A_affect"))
            emotion_id = self.emotion2id.get(a, -1) if a is not None else -1

            # S_strategy is multi-label (e.g. [격려, 동조]).
            # strategy_scope=session: axes.S_strategy; utterance: last listener turn tags.
            strategies = (row.get("axes") or {}).get("S_strategy") or []
            if not isinstance(strategies, list):
                strategies = [strategies] if strategies is not None else []
            if getattr(self, "strategy_scope", "utterance") == "utterance":
                for u in reversed(row.get("dialogue") or []):
                    if u.get("role") == "listener" and u.get("strategies"):
                        strategies = u["strategies"]
                        break
            strategy_ids_multi = []
            for s in strategies:
                sid = self.strategy2id.get(str(s), -1)
                if sid >= 0:
                    strategy_ids_multi.append(sid)
            # primary: rarest class among positives (reduces majority collapse vs always first)
            if strategy_ids_multi:
                # frequency proxy from label order is unavailable here; use last as response-facing
                primary_s_id = strategy_ids_multi[0]
            else:
                primary_s_id = -1

            r = (row.get("axes") or {}).get("R_relation")
            relation_id = self.relation2id.get(str(r), -1) if r is not None else -1

            self.rows.append(
                {
                    "id": row.get("id"),
                    "prompt": prompt,
                    "target": target,
                    "emotion_id": emotion_id,
                    "strategy_id": primary_s_id,
                    "strategy_ids_multi": strategy_ids_multi,
                    "relation_id": relation_id,
                    "axes": row.get("axes") or {},
                    "source": row.get("source"),
                    "lang": row.get("lang"),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class EmpathyCollator:
    """Collate prompt/target into causal-LM batches (+ axis ids)."""

    def __init__(
        self,
        tokenizer,
        *,
        max_length: int = 512,
        response_prefix: str = " ",
        n_strategies: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_prefix = response_prefix
        self.n_strategies = n_strategies
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        prompts = [f["prompt"] for f in features]
        targets = [self.response_prefix + (f["target"] or "") for f in features]
        eos = self.tokenizer.eos_token or ""

        # Reserve room for targets so labels are not entirely masked (-100),
        # which yields NaN causal-LM loss (common with long KO prompts on gpt2).
        target_tok = self.tokenizer(
            [t + eos for t in targets],
            padding=False,
            truncation=True,
            max_length=max(32, self.max_length // 3),
            add_special_tokens=False,
        )
        prompt_budget = [
            max(16, self.max_length - len(ids)) for ids in target_tok["input_ids"]
        ]
        prompt_tok_list = []
        for p, budget in zip(prompts, prompt_budget):
            pt = self.tokenizer(
                p,
                truncation=True,
                max_length=budget,
                add_special_tokens=True,
                return_tensors=None,
            )
            prompt_tok_list.append(pt["input_ids"])

        input_ids = []
        labels = []
        for p_ids, t_ids in zip(prompt_tok_list, target_tok["input_ids"]):
            ids = (p_ids + t_ids)[: self.max_length]
            lab = ([-100] * len(p_ids) + t_ids)[: self.max_length]
            # if still no supervised tokens, supervise last token
            if all(x == -100 for x in lab) and ids:
                lab[-1] = ids[-1]
            input_ids.append(ids)
            labels.append(lab)

        # pad
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(x) for x in input_ids)
        padded_ids = []
        padded_labels = []
        attn = []
        for ids, lab in zip(input_ids, labels):
            pad_n = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_n)
            padded_labels.append(lab + [-100] * pad_n)
            attn.append([1] * len(ids) + [0] * pad_n)

        out: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "emotion_ids": torch.tensor(
                [f.get("emotion_id", -1) for f in features], dtype=torch.long
            ),
            "strategy_ids": torch.tensor(
                [f.get("strategy_id", -1) for f in features], dtype=torch.long
            ),
            "relation_ids": torch.tensor(
                [f.get("relation_id", -1) for f in features], dtype=torch.long
            ),
        }
        # Multi-label strategy targets [B, C] when n_strategies known
        n_s = self.n_strategies
        if n_s is None:
            # infer from any multi list present
            for f in features:
                multi = f.get("strategy_ids_multi")
                if multi:
                    n_s = max(n_s or 0, max(multi) + 1)
        if n_s is not None and n_s > 0:
            multihot = torch.zeros(len(features), n_s, dtype=torch.float32)
            for i, f in enumerate(features):
                multi = f.get("strategy_ids_multi") or []
                if not multi and f.get("strategy_id", -1) >= 0:
                    multi = [int(f["strategy_id"])]
                for sid in multi:
                    if 0 <= int(sid) < n_s:
                        multihot[i, int(sid)] = 1.0
            out["strategy_multihot"] = multihot
        return out
