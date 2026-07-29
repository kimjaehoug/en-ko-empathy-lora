#!/usr/bin/env python3
"""Build unified processed JSONL records for EN/KO empathy corpora."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

csv.field_size_limit(10_000_000)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)})")


def record(
    *,
    rid: str,
    source: str,
    split: str,
    lang: str,
    situation: str | None,
    dialogue: list[dict],
    A,
    C,
    S,
    R,
    meta: dict | None = None,
    situation_en: str | None = None,
    situation_ko: str | None = None,
) -> dict:
    return {
        "id": rid,
        "source": source,
        "split": split,
        "lang": lang,
        "situation": situation,
        "situation_en": situation_en,
        "situation_ko": situation_ko,
        "dialogue": dialogue,
        "axes": {
            "A_affect": A,
            "C_cognition": C,
            "S_strategy": S,
            "R_relation": R,
        },
        "meta": meta or {},
    }


def fix_ed_text(s: str) -> str:
    return (s or "").replace("_comma_", ",")


def build_empathetic_dialogues() -> dict[str, list[dict]]:
    base = RAW / "empathetic_dialogues" / "empatheticdialogues"
    out: dict[str, list[dict]] = {}
    split_map = {"train": "train", "valid": "valid", "test": "test"}
    for split_file, split in split_map.items():
        path = base / f"{split_file}.csv"
        by_conv: dict[str, list[dict]] = defaultdict(list)
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                by_conv[row["conv_id"]].append(row)
        rows = []
        for conv_id, utts in by_conv.items():
            utts = sorted(utts, key=lambda r: int(r["utterance_idx"]))
            emotion = fix_ed_text(utts[0].get("context", ""))
            situation = fix_ed_text(utts[0].get("prompt", ""))
            dialogue = []
            for i, u in enumerate(utts):
                # ED: odd turns speaker (0-index even?), utterance_idx starts at 1
                # typically idx 1,3,5 speaker; 2,4,6 listener
                idx = int(u["utterance_idx"])
                role = "speaker" if idx % 2 == 1 else "listener"
                dialogue.append(
                    {
                        "utt_id": f"{conv_id}:{idx}",
                        "role": role,
                        "text": fix_ed_text(u["utterance"]),
                        "strategies": None,
                    }
                )
            rows.append(
                record(
                    rid=f"ed:{conv_id}",
                    source="empathetic_dialogues",
                    split=split,
                    lang="en",
                    situation=situation,
                    dialogue=dialogue,
                    A=emotion,
                    C=situation,
                    S=[],
                    R=None,
                    meta={"n_utterances": len(dialogue)},
                )
            )
        out[split] = rows
    return out


def build_kor_ed() -> dict[str, list[dict]]:
    base = RAW / "kor_empathetic_dialogues"
    out: dict[str, list[dict]] = {}
    for split, fname in [
        ("train", "train.jsonl"),
        ("valid", "validation.jsonl"),
        ("test", "test.jsonl"),
    ]:
        rows = []
        with (base / fname).open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                did = obj["dialogue_id"]
                dialogue = []
                for u in obj["dialogue"]:
                    idx = u["utter_idx"]
                    # same parity as ED: 0 speaker? kor uses 0-index; user_id 0/1
                    role = "speaker" if u["user_id"] % 2 == 0 else "listener"
                    dialogue.append(
                        {
                            "utt_id": f"kor_ed:{did}:{idx}",
                            "role": role,
                            "text": u["utter"],
                            "strategies": None,
                        }
                    )
                rows.append(
                    record(
                        rid=f"kor_ed:{did}",
                        source="kor_empathetic_dialogues",
                        split=split,
                        lang="ko",
                        situation=obj.get("situation"),
                        dialogue=dialogue,
                        A=obj.get("emotion"),
                        C=obj.get("situation"),
                        S=[],
                        R=None,
                        meta={"note": "machine_translated_baseline"},
                    )
                )
        out[split] = rows
    return out


def build_koed() -> dict[str, list[dict]]:
    path = RAW / "KoED" / "data" / "KoED.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for obj in data:
        conv_id = obj["conv_id"]
        dialogue = []
        for u in obj["dialogue"]:
            idx = u["utter_idx"]
            # user_id alternates; first speaker-like
            uid = u.get("user_id")
            if uid is None:
                role = "speaker" if idx % 2 == 1 else "listener"
            else:
                # keep relative: min user is speaker side for this dataset convention
                role = "speaker" if idx % 2 == 1 else "listener"
            dialogue.append(
                {
                    "utt_id": f"koed:{conv_id}:{idx}",
                    "role": role,
                    "text": u.get("ko_utter") or u.get("utter"),
                    "text_en": u.get("utter"),
                    "text_ko": u.get("ko_utter"),
                    "strategies": None,
                }
            )
        rows.append(
            record(
                rid=f"koed:{conv_id}",
                source="koed",
                split="test",  # evaluation-only recommended
                lang="ko",
                situation=obj.get("ko_situation") or obj.get("situation"),
                situation_en=obj.get("situation"),
                situation_ko=obj.get("ko_situation"),
                dialogue=dialogue,
                A=obj.get("emotion"),
                C=obj.get("ko_situation") or obj.get("situation"),
                S=[],
                R=None,
                meta={"eval_only": True, "parallel": True},
            )
        )
    return {"test": rows}


def build_aihub() -> dict[str, list[dict]]:
    root = RAW / "aihub_empathy" / "046.공감형 대화" / "01-1.정식개방데이터"
    # Use labeling folders only (TL_/VL_), one json = one session
    files = [
        p
        for p in root.rglob("Empathy_*.json")
        if ("/TL_" in str(p)) or ("/VL_" in str(p))
    ]
    rows_by_split: dict[str, list[dict]] = {"train": [], "valid": []}
    seen = set()
    for p in files:
        split = "train" if "/Training/" in str(p) else "valid"
        obj = json.loads(p.read_text(encoding="utf-8"))
        info = obj["info"]
        rid = f"aihub:{info.get('id') or p.stem}"
        if rid in seen:
            continue
        seen.add(rid)
        dialogue = []
        utt_strategies = []
        for u in obj.get("utterances") or []:
            strat = u.get("listener_empathy") or None
            if strat:
                utt_strategies.extend(strat)
            dialogue.append(
                {
                    "utt_id": u.get("utterance_id"),
                    "role": u.get("role"),
                    "text": u.get("text"),
                    "strategies": strat,
                }
            )
        session_S = info.get("listener_behavior") or []
        # prefer utterance-level unique strategies if present else session
        S = sorted(set(utt_strategies)) if utt_strategies else list(session_S)
        rows_by_split[split].append(
            record(
                rid=rid,
                source="aihub_empathy",
                split=split,
                lang="ko",
                situation=info.get("situation"),
                dialogue=dialogue,
                A=info.get("speaker_emotion"),
                C=info.get("situation"),
                S=S,
                R=info.get("relation"),
                meta={
                    "grade": (info.get("evaluation") or {}).get("grade"),
                    "avg_rating": (info.get("evaluation") or {}).get("avg_rating"),
                    "listener_behavior_session": session_S,
                    "file": str(p.relative_to(RAW)),
                },
            )
        )
    return rows_by_split


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"axes": ["A_affect", "C_cognition", "S_strategy", "R_relation"], "files": {}}

    print("== EmpatheticDialogues ==")
    ed = build_empathetic_dialogues()
    for split, rows in ed.items():
        path = OUT / f"ed_{split}.jsonl"
        write_jsonl(path, rows)
        manifest["files"][f"ed_{split}"] = len(rows)

    print("== KorEmpatheticDialogues ==")
    kor = build_kor_ed()
    for split, rows in kor.items():
        path = OUT / f"kor_ed_{split}.jsonl"
        write_jsonl(path, rows)
        manifest["files"][f"kor_ed_{split}"] = len(rows)

    print("== KoED ==")
    koed = build_koed()
    for split, rows in koed.items():
        path = OUT / f"koed_{split}.jsonl"
        write_jsonl(path, rows)
        manifest["files"][f"koed_{split}"] = len(rows)

    print("== AI Hub ==")
    aihub = build_aihub()
    for split, rows in aihub.items():
        path = OUT / f"aihub_{split}.jsonl"
        write_jsonl(path, rows)
        manifest["files"][f"aihub_{split}"] = len(rows)

    # combined KO train view (aihub only for S/R supervision)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nmanifest:", json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
