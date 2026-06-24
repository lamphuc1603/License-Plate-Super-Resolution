"""
OCR1 + OCR2 combination logic for final text prediction.
"""

import torch

from config import (
    CHARACTERS, CHAR_DICT, NUM_CLASSES,
    OCR1_WEIGHT, FILL_THRESH,
)


def decide_final_text(raw_text, pos_display, crnn_decoded, crnn_char_probs,
                      expected_len):
    """
    Combine OCR1 (TrOCR voting) and OCR2 (CRNN) predictions per position.

    Rules:
      - raw != '_' → combined = OCR1_WEIGHT*OCR1 + (1-OCR1_WEIGHT)*OCR2 → argmax
      - raw == '_' → fill only if max(combined) > FILL_THRESH
    """
    clean = raw_text.replace("-", "")
    length_ok = len(crnn_decoded) == expected_len
    result = []

    for i, raw_char in enumerate(clean[:expected_len]):
        # OCR1 distribution
        ocr1 = torch.zeros(NUM_CLASSES)
        if raw_char != "_" and raw_char in CHAR_DICT:
            ocr1[CHAR_DICT[raw_char]] = 1.0
        elif i in pos_display and pos_display[i]:
            for ch, prob in pos_display[i].items():
                if ch in CHAR_DICT:
                    ocr1[CHAR_DICT[ch]] = prob
            ocr1 /= ocr1.sum() + 1e-8
        else:
            ocr1.fill_(1.0 / NUM_CLASSES)

        # OCR2 distribution
        ocr2 = torch.zeros(NUM_CLASSES)
        if length_ok and i < len(crnn_char_probs):
            for ch, prob in crnn_char_probs[i].items():
                if ch in CHAR_DICT:
                    ocr2[CHAR_DICT[ch]] = prob
            ocr2 = ocr2 / (ocr2.sum() + 1e-8)
        else:
            ocr2.fill_(1.0 / NUM_CLASSES)

        combined = OCR1_WEIGHT * ocr1 + (1 - OCR1_WEIGHT) * ocr2
        pred_idx = combined.argmax().item()
        pred_char = CHARACTERS[pred_idx]
        pred_conf = combined[pred_idx].item()
        decision = pred_char if (raw_char != "_" or pred_conf > FILL_THRESH) else "_"
        result.append(decision)

    return "".join(result)
