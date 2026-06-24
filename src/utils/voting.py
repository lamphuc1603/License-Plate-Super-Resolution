"""
TrOCR ensemble soft-voting and label vector generation for TPGSR.
"""

import torch
from collections import defaultdict
from PIL import Image

from config import (
    DEVICE, CHARACTERS, CHAR_DICT, NUM_CLASSES,
    PLATE_REGEXES, PLATE_PATTERNS, PLATE_LENGTHS, VALID_LETTERS,
    MAX_TARGET_LEN, NUM_BEAMS, TOP_K,
    UNDERSCORE_THRESH, INVALID_THRESH,
)


def predict_with_soft_voting(image_paths, models, processor, num_beams=NUM_BEAMS):
    """
    TrOCR ensemble soft-voting across multiple models × multiple frames.

    Returns:
        raw_text:       voted plate string (may contain '_' for uncertain positions)
        best_type:      detected plate type (1, 2, or 3)
        pos_display:    per-position probability distributions
        top_candidates: top-K candidate texts with scores
        meta:           diagnostic metadata
    """
    score_sum = defaultdict(float)
    count_board = defaultdict(int)

    with torch.no_grad():
        for model in models:
            for img_path in image_paths:
                img = Image.open(img_path).convert("RGB")
                pv = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
                out = model.generate(
                    pv,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_length=MAX_TARGET_LEN,
                )
                decoded = processor.batch_decode(
                    out.sequences, skip_special_tokens=True
                )
                for text, sc in zip(decoded, out.sequences_scores.tolist()):
                    text = text.strip().upper()
                    score_sum[text] += sc
                    count_board[text] += 1

    if not score_sum:
        return "_" * 8, 1, {}, [], {
            "pool_size": 0, "type_votes": {}, "total_unique": 0, "typed_total": 0,
        }

    avg_score = {t: score_sum[t] / count_board[t] for t in score_sum}
    ranked = sorted(avg_score.items(), key=lambda x: x[1], reverse=True)
    top_k = ranked[:TOP_K]

    # ── Identify plate type ──────────────────────────────────────────
    type_count = defaultdict(int)
    for text, _ in top_k:
        for pt, regex in PLATE_REGEXES.items():
            if regex.match(text):
                type_count[pt] += 1
                break

    if sum(type_count.values()) < 3:
        type_count_full = defaultdict(int)
        for text, _ in ranked:
            for pt, regex in PLATE_REGEXES.items():
                if regex.match(text):
                    type_count_full[pt] += 1
                    break
        if sum(type_count_full.values()) > 10:
            type_count = type_count_full
        else:
            type_count = {}

    # ── No type found Branch ───────────────────────────────
    if not type_count:
        len_count = defaultdict(int)
        for text, _ in ranked:
            len_count[len(text)] += 1
        most_common_len = max(len_count, key=len_count.get)

        if most_common_len == 7:
            best_type = 3
        elif most_common_len == 8:
            dl_votes = defaultdict(int)
            for text, _ in ranked:
                if len(text) != 8:
                    continue
                for pt in (1, 2):
                    pat = PLATE_PATTERNS[pt]
                    dl_votes[pt] += sum(
                        (c.isdigit() and p == "D") or (c in VALID_LETTERS and p == "L")
                        for c, p in zip(text, pat)
                    )
            best_type = max(dl_votes, key=dl_votes.get) if dl_votes else 1
        else:
            best_type = 1
            return (
                "_" * PLATE_LENGTHS[best_type], best_type, {},
                [(t, sc, count_board[t]) for t, sc in top_k],
                {
                    "pool_size": 0, "type_votes": {},
                    "total_unique": len(score_sum), "typed_total": 0,
                },
            )

        # Fallback: Vote on invalid pool based on length
        target_len = PLATE_LENGTHS[best_type]
        invalid_pool = [(t, sc) for t, sc in ranked if len(t) == target_len]
        vote_pattern = PLATE_PATTERNS[best_type]
        forced_under = set() if best_type == 3 else {3}

        pos_count_inv = defaultdict(lambda: defaultdict(int))
        for text, _ in invalid_pool:
            for i, c in enumerate(text):
                pos_count_inv[i][c] += 1

        result_inv = []
        for i, p in enumerate(vote_pattern):
            if i in forced_under:
                result_inv.append("_")
                continue
            all_chars = pos_count_inv.get(i, {})
            total_count = sum(all_chars.values()) or 1
            chosen = "_"
            for ch, cnt in sorted(all_chars.items(), key=lambda x: -x[1]):
                pct = cnt / total_count * 100
                is_valid = (p == "D" and ch.isdigit()) or (p == "L" and ch in VALID_LETTERS)
                if is_valid and pct >= INVALID_THRESH:
                    chosen = ch
                    break
            result_inv.append(chosen)

        return (
            "".join(result_inv), best_type, {},
            [(t, sc, count_board[t]) for t, sc in top_k],
            {
                "pool_size": len(invalid_pool), "type_votes": {},
                "total_unique": len(score_sum), "typed_total": 0,
            },
        )

    # ── Typed Branch: vote per-position ───────────────────────────────
    best_type = max(type_count, key=type_count.get)
    pattern = PLATE_PATTERNS[best_type]
    regex = PLATE_REGEXES[best_type]
    typed_ranked = [(t, sc) for t, sc in ranked if regex.match(t)]

    pos_count = defaultdict(lambda: defaultdict(int))
    for text, _ in typed_ranked:
        for i, c in enumerate(text):
            pos_count[i][c] += 1

    result, pos_display = [], {}
    for i, p in enumerate(pattern):
        all_chars = pos_count.get(i, {})
        total_count = sum(all_chars.values()) or 1
        sorted_chars = sorted(all_chars.items(), key=lambda x: -x[1])
        chosen, chosen_pct = None, 0.0
        for ch, cnt in sorted_chars:
            if (p == "D" and ch.isdigit()) or (p == "L" and ch in VALID_LETTERS):
                chosen = ch
                chosen_pct = cnt / total_count * 100
                break
        if chosen is None or chosen_pct < UNDERSCORE_THRESH:
            chosen = "_"
        result.append(chosen)
        pos_display[i] = {ch: cnt / total_count for ch, cnt in sorted_chars}

    return (
        "".join(result), best_type, pos_display,
        [(t, sc, count_board[t]) for t, sc in top_k],
        {
            "pool_size": len(typed_ranked[:TOP_K]),
            "type_votes": dict(type_count),
            "total_unique": len(score_sum),
            "typed_total": len(typed_ranked),
        },
    )


def voting_to_label_vecs(raw_text, pos_display, device=DEVICE):
    """
    Convert OCR1 voting result → label_vecs tensor [1, C, 1, 8] for TPGSR.

    - Confident chars → one-hot
    - Uncertain chars → use voting distribution
    - Unknown/padding → uniform distribution
    """
    lv = torch.zeros(8, NUM_CLASSES)
    clean = raw_text.replace("-", "")
    for i, char in enumerate(clean[:8]):
        if i in pos_display and pos_display[i]:
            for ch, prob in pos_display[i].items():
                if ch in CHAR_DICT:
                    lv[i, CHAR_DICT[ch]] = prob
        elif char != "_" and char in CHAR_DICT:
            lv[i, CHAR_DICT[char]] = 1.0
        else:
            lv[i] = 1.0 / NUM_CLASSES
    for i in range(min(len(clean), 8), 8):
        lv[i] = 1.0 / NUM_CLASSES
    lv = lv / (lv.sum(-1, keepdim=True) + 1e-8)
    return lv.T.unsqueeze(0).unsqueeze(2).to(device)  # [1, C, 1, 8]
