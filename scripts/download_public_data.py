#!/usr/bin/env python3
"""Download public EN/KO empathy corpora into data/raw/."""

from __future__ import annotations

import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

ED_URL = (
    "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
)


def save_hf_split(ds_dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for split, ds in ds_dict.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        meta[split] = {"n": len(ds), "path": str(path.relative_to(ROOT))}
        print(f"  wrote {path} ({len(ds)} rows)")
    (out_dir / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def download_empathetic_dialogues() -> None:
    print("== EmpatheticDialogues (ParlAI tarball) ==")
    out = RAW / "empathetic_dialogues"
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / "empatheticdialogues.tar.gz"
    extracted = out / "empatheticdialogues"
    if extracted.exists() and any(extracted.glob("*.csv")):
        print(f"  exists: {extracted}")
        return
    print(f"  fetching {ED_URL}")
    req = urllib.request.Request(ED_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, tar_path.open("wb") as f:
        f.write(resp.read())
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(out)
    for split in ("train", "valid", "test"):
        csv = extracted / f"{split}.csv"
        n = sum(1 for _ in csv.open(encoding="utf-8", errors="ignore")) - 1
        print(f"  {split}.csv rows≈{n}")


def download_kor_ed() -> None:
    from datasets import load_dataset

    print("== KorEmpatheticDialogues ==")
    ds = load_dataset("passing2961/KorEmpatheticDialogues")
    save_hf_split(ds, RAW / "kor_empathetic_dialogues")


def clone_koed() -> None:
    print("== KoED (git) ==")
    dest = RAW / "KoED"
    if dest.exists():
        print(f"  exists: {dest}")
        return
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/KUNLP/KoED.git",
            str(dest),
        ]
    )
    print(f"  cloned: {dest}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    clone_koed()
    download_empathetic_dialogues()
    download_kor_ed()
    print("\nDone.")
    print("AI Hub 공감형 대화는 수동 신청 후 data/raw/aihub_empathy/ 에 두세요.")
    print("https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71305")


if __name__ == "__main__":
    main()
