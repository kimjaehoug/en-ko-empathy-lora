"""Readable training progress logs + progress.json for tail -f."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _pct(step: int, total: int | None) -> str:
    if not total or total <= 0:
        return "?"
    return f"{100.0 * step / total:.1f}%"


def _fmt_acc(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{100.0 * v:.2f}%"


def _fmt_loss(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}"


class TrainLogger:
    """Console + progress.json logger for long GPU runs."""

    def __init__(
        self,
        *,
        stage: str,
        out_dir: Path | str,
        total_steps: int | None = None,
        log_every: int = 10,
    ) -> None:
        self.stage = stage
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.total_steps = total_steps
        self.log_every = max(1, int(log_every))
        self.progress_path = self.out_dir / "progress.json"
        self._t0 = time.time()
        self._state: dict[str, Any] = {
            "stage": stage,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "total_steps": total_steps,
            "global_step": 0,
            "status": "running",
        }
        self._write_progress()

    def _write_progress(self) -> None:
        self._state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._state["elapsed_sec"] = round(time.time() - self._t0, 1)
        self.progress_path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update(self, **kwargs: Any) -> None:
        self._state.update(kwargs)
        self._write_progress()

    def banner(self, msg: str) -> None:
        line = f"[{self.stage}] {msg}"
        print(line, flush=True)
        self._state["banner"] = msg
        self._write_progress()

    def set_total_steps(self, total_steps: int) -> None:
        self.total_steps = int(total_steps)
        self._state["total_steps"] = self.total_steps
        self._write_progress()

    def log_train(
        self,
        step: int,
        *,
        loss: float,
        lm_loss: float | None = None,
        extra_losses: dict[str, float | None] | None = None,
        train_acc: dict[str, float | None] | None = None,
        force: bool = False,
    ) -> None:
        self._state["global_step"] = step
        self._state["train"] = {
            "loss": loss,
            "lm_loss": lm_loss,
            **(extra_losses or {}),
            **(train_acc or {}),
        }
        self._write_progress()

        if not force and step % self.log_every != 0 and step != 1:
            return

        parts = [
            f"[{self.stage}]",
            f"step {step}/{self.total_steps or '?'} ({_pct(step, self.total_steps)})",
            f"loss={_fmt_loss(loss)}",
        ]
        if lm_loss is not None:
            parts.append(f"lm={_fmt_loss(lm_loss)}")
        for k, v in (extra_losses or {}).items():
            if v is not None:
                parts.append(f"{k}={_fmt_loss(v)}")
        for k, v in (train_acc or {}).items():
            if v is not None:
                parts.append(f"{k}={_fmt_acc(v)}")
        if self._state.get("elapsed_sec") is not None:
            parts.append(f"elapsed={self._state['elapsed_sec']:.0f}s")
        print(" | ".join(parts), flush=True)

    def log_eval(self, step: int, metrics: dict[str, Any], *, split: str = "valid") -> None:
        self._state["global_step"] = step
        self._state[f"eval_{split}"] = metrics
        self._write_progress()

        acc_keys = [k for k in metrics if k.endswith("_acc")]
        acc_parts = [f"{k.replace('_acc', '').upper()}={_fmt_acc(metrics[k])}" for k in acc_keys]
        loss_parts = []
        for k in ("loss", "lm_loss", "emotion_loss", "strategy_loss", "relation_loss"):
            if metrics.get(k) is not None:
                loss_parts.append(f"{k}={_fmt_loss(metrics[k])}")

        print(
            f"[{self.stage}] EVAL @{step} ({split}) | "
            + " | ".join(acc_parts + loss_parts),
            flush=True,
        )

    def log_final(self, metrics: dict[str, Any], *, extra: dict[str, Any] | None = None) -> None:
        self._state["status"] = "done"
        self._state["final_eval"] = metrics
        if extra:
            self._state.update(extra)
        self._write_progress()

        acc_keys = [k for k in metrics if k.endswith("_acc")]
        acc_parts = [f"{k.replace('_acc', '').upper()}={_fmt_acc(metrics[k])}" for k in acc_keys]
        print(
            f"[{self.stage}] FINAL | " + " | ".join(acc_parts),
            flush=True,
        )
        if extra:
            for k, v in extra.items():
                print(f"[{self.stage}] {k}={v}", flush=True)

    def fail(self, msg: str) -> None:
        self._state["status"] = "failed"
        self._state["error"] = msg
        self._write_progress()
        print(f"[{self.stage}] ERROR: {msg}", flush=True)


def batch_accuracy(
    logits: Any,
    ids: Any,
    *,
    name: str,
) -> dict[str, float | None]:
    """Single-batch classification accuracy (skip ids < 0)."""
    import torch

    if ids is None:
        return {f"train_{name}_acc": None}
    valid = ids >= 0
    if not valid.any():
        return {f"train_{name}_acc": None}
    pred = logits[valid].argmax(dim=-1)
    acc = float((pred == ids[valid]).float().mean().item())
    return {f"train_{name}_acc": acc}


def estimate_total_steps(
    *,
    n_train: int,
    batch_size: int,
    grad_accum: int,
    num_epochs: int,
    max_steps: int | None,
) -> int:
    steps_per_epoch = max((n_train + batch_size - 1) // batch_size // grad_accum, 1)
    total = steps_per_epoch * num_epochs
    if max_steps is not None:
        return min(int(max_steps), total)
    return total
