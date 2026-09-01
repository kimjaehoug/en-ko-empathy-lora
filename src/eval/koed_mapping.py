"""KoED ↔ ED emotion mapping for held-out cultural eval (KoED paper protocol)."""
from __future__ import annotations

# Shared ED emotions evaluated in KoED paper (Lee et al. 2025) — 8-class subset
KOED_SHARED_EMOTIONS = [
    "afraid",
    "angry",
    "annoyed",
    "anxious",
    "grateful",
    "joyful",
    "sad",
    "surprised",
]

# Map fine ED labels → shared bucket (identity for shared; nearest for others)
ED_TO_SHARED = {
    "afraid": "afraid",
    "terrified": "afraid",
    "anxious": "anxious",
    "apprehensive": "anxious",
    "angry": "angry",
    "furious": "angry",
    "annoyed": "annoyed",
    "grateful": "grateful",
    "joyful": "joyful",
    "excited": "joyful",
    "hopeful": "joyful",
    "sad": "sad",
    "lonely": "sad",
    "devastated": "sad",
    "disappointed": "sad",
    "surprised": "surprised",
    "sentimental": "sad",
    "nostalgic": "sad",
}


def koed_primary_emotion(affect: list[str] | str | None) -> str | None:
    """Pick primary KoED emotion label (first listed) mapped to shared set."""
    if affect is None:
        return None
    if isinstance(affect, str):
        raw = affect
    elif isinstance(affect, list) and affect:
        raw = str(affect[0])
    else:
        return None
    raw = raw.strip().lower()
    return ED_TO_SHARED.get(raw, raw if raw in KOED_SHARED_EMOTIONS else None)


def koed_multilabel_hits(affect: list[str] | str | None) -> set[str]:
    if affect is None:
        return set()
    items = [affect] if isinstance(affect, str) else [str(x) for x in affect]
    out = set()
    for raw in items:
        mapped = ED_TO_SHARED.get(raw.strip().lower())
        if mapped:
            out.add(mapped)
    return out
