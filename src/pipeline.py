"""
InferencePipeline — orchestrates the full XLPSR inference workflow.

Usage:
    pipeline = InferencePipeline()   # loads all models
    pipeline.run()                   # reads INPUT_DIR, writes OUTPUT_DIR
"""

import os
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from config import (
    DEVICE, INPUT_DIR, OUTPUT_DIR, CKPT_DIR,
    TPGSR_CKPT, CRNN_CKPT, TROCR_DIRS,
    NUM_CLASSES, PLATE_LENGTHS,
    LR_H, LR_W, HR_H, HR_W,
)
from model.rrdb import RRDB_TL_LP
from model.crnn import CRNN
from utils.voting import predict_with_soft_voting, voting_to_label_vecs
from utils.combine import decide_final_text
from utils.image import tensor_to_np


class InferencePipeline:
    """
    Full inference pipeline:
        TrOCR ensemble (OCR1) → TPGSR (SR) → CRNN (OCR2) → Combine → Output
    """

    def __init__(self):
        """Load all models into memory."""
        print(f"Device: {DEVICE}")

        self.tpgsr = self._load_tpgsr()
        self.crnn = self._load_crnn()
        self.processor, self.ensemble_models = self._load_trocr()

        print(f"\nReady: {len(self.ensemble_models)}x TrOCR + TPGSR + CRNN")

    # ─────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────

    def _load_tpgsr(self):
        model = RRDB_TL_LP(
            scale_factor=2, width=HR_W, height=HR_H,
            in_nc=3, nf=64, nb=8, gc=32,
            text_emb=NUM_CLASSES, out_text_channels=64,
        ).to(DEVICE)
        ckpt = torch.load(str(TPGSR_CKPT), map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"TPGSR  epoch={ckpt['epoch']}  best_psnr={ckpt['best_psnr']:.2f} dB")
        del ckpt
        return model

    def _load_crnn(self):
        model = CRNN().to(DEVICE)
        ckpt = torch.load(str(CRNN_CKPT), map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"CRNN   epoch={ckpt['epoch']}")
        del ckpt
        return model

    def _load_trocr(self):
        processor = TrOCRProcessor.from_pretrained(str(CKPT_DIR / "trocr_processor"))
        models = []
        for d in TROCR_DIRS:
            ckpts = sorted(d.glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[-1]))
            load_from = ckpts[-1] if ckpts else d
            m = VisionEncoderDecoderModel.from_pretrained(str(load_from)).to(DEVICE)
            m.eval()
            models.append(m)
            print(f"TrOCR  {load_from.name}")
        return processor, models

    # ─────────────────────────────────────────────────────────────────
    # Sequence discovery
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _discover_sequences():
        """
        Read sequence list from all_sequences.csv if available,
        otherwise fallback to listing subdirectories in INPUT_DIR.
        """
        csv_path = INPUT_DIR / "all_sequences.csv"

        if csv_path.exists():
            seq_names = []
            with open(str(csv_path), "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Support column names: sequence_id, seq_id, id, or first column
                    name = (
                        row.get("sequence_id")
                        or row.get("seq_id")
                        or row.get("id")
                        or list(row.values())[0]
                    )
                    if name:
                        seq_names.append(name.strip())
            seq_dirs = [INPUT_DIR / name for name in seq_names
                        if (INPUT_DIR / name).is_dir()]
            print(f"Loaded {len(seq_dirs)} sequences from {csv_path.name}")
        else:
            seq_dirs = sorted(
                p for p in INPUT_DIR.iterdir()
                if p.is_dir()
            )
            print(f"No CSV found — discovered {len(seq_dirs)} sequence directories")

        return seq_dirs

    # ─────────────────────────────────────────────────────────────────
    # Single-sequence inference
    # ─────────────────────────────────────────────────────────────────

    def infer_one_sequence(self, seq_dir):
        """
        Full pipeline for one sequence:
            frames → OCR1 voting → TPGSR → noise injection → CRNN → combine

        Returns:
            (final_text, sr_tensor) or None if no images found.
        """
        frames = sorted(seq_dir.glob("*.png")) + sorted(seq_dir.glob("*.jpg"))
        if not frames:
            return None

        # ── Step 1: OCR1 ensemble voting ─────────────────────────────
        raw_text, best_type, pos_display, _, _ = \
            predict_with_soft_voting(frames, self.ensemble_models, self.processor)

        expected_len = PLATE_LENGTHS[best_type]
        clean = raw_text.replace("-", "")
        is_full_under = all(c == "_" for c in clean)

        # ── Step 2: Load LR (middle frame) ───────────────────────────
        lr_img = cv2.cvtColor(cv2.imread(str(frames[min(4, len(frames) - 1)])), cv2.COLOR_BGR2RGB)
        lr_img = cv2.resize(lr_img, (LR_W, LR_H), interpolation=cv2.INTER_CUBIC)
        lr_t = (
            torch.from_numpy(lr_img.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            # ── Step 3: TPGSR ────────────────────────────────────────
            lv = voting_to_label_vecs(raw_text, pos_display)
            sr = self.tpgsr(lr_t, lv)  # [1, 3, HR_H, HR_W]

            if is_full_under:
                return clean, sr[0]

            # ── Step 4: Noise injection at underscore positions ──────
            self._inject_noise(sr, clean, expected_len)

            # ── Step 4b: CRNN reads SR ───────────────────────────────
            sr_in = F.interpolate(
                sr.clamp(0, 1), size=(HR_H, HR_W),
                mode="bilinear", align_corners=False,
            )
            log_probs = self.crnn(sr_in)
            crnn_decoded, crnn_char_probs = self.crnn.decode_with_probs(log_probs)[0]

            # ── Step 5: Combine OCR1 + OCR2 ──────────────────────────
            final_text = decide_final_text(
                raw_text, pos_display, crnn_decoded, crnn_char_probs,
                expected_len,
            )

        return final_text, sr[0]

    @staticmethod
    def _inject_noise(sr, clean, expected_len):
        """
        Apply Gaussian blur + grain noise at underscore positions
        to help CRNN focus on uncertain character regions.
        """
        LEFT_OFFSET = 11
        VERTICAL_MARGIN = int(HR_H * 0.2)
        y0 = VERTICAL_MARGIN
        y1 = HR_H - VERTICAL_MARGIN
        char_w = (HR_W - 2 * LEFT_OFFSET) // expected_len

        for i, char in enumerate(clean[:expected_len]):
            if char != "_":
                continue

            x0 = max(0, LEFT_OFFSET + i * char_w)
            x1 = min(HR_W, LEFT_OFFSET + (i + 1) * char_w)

            region = sr[:, :, y0:y1, x0:x1]

            # Gaussian blur
            blurred_region = TF.gaussian_blur(region, kernel_size=[3, 3], sigma=[1, 1])

            # Feathering mask
            mask = torch.ones((1, 1, y1 - y0, x1 - x0), device=sr.device)
            fade_pixels = 6
            if (x1 - x0) > 2 * fade_pixels:
                ramp = torch.linspace(0, 1, fade_pixels, device=sr.device)
                mask[..., :fade_pixels] = ramp
                mask[..., -fade_pixels:] = ramp.flip(0)

            # Grain noise
            noise = torch.randn_like(region) * 0.02

            # Blend
            sr[:, :, y0:y1, x0:x1] = (blurred_region + noise) * mask + region * (1 - mask)
            sr.clamp_(0, 1)

    # ─────────────────────────────────────────────────────────────────
    # Main run loop
    # ─────────────────────────────────────────────────────────────────

    def run(self):
        """
        Discover sequences from INPUT_DIR, run inference on each,
        and save results (sr.png + text.txt) to OUTPUT_DIR.
        """
        seq_dirs = self._discover_sequences()
        print(f"\nRunning on {len(seq_dirs)} sequences...\n")

        os.makedirs(str(OUTPUT_DIR), exist_ok=True)

        for seq_dir in tqdm(seq_dirs, desc="Inference"):
            result = self.infer_one_sequence(seq_dir)

            if result is None:
                print(f"[WARN] No images in {seq_dir.name}")
                continue

            final_text, sr_tensor = result

            # Create output directory
            sample_out = OUTPUT_DIR / seq_dir.name
            os.makedirs(str(sample_out), exist_ok=True)

            # Save sr.png
            sr_np = tensor_to_np(sr_tensor)
            sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(sample_out / "sr.png"), sr_bgr)

            # Save text.txt
            with open(str(sample_out / "text.txt"), "w", encoding="utf-8") as f:
                f.write(final_text)

            print(f"[OK] {seq_dir.name}: {final_text}")

        print(f"\nDone. Output saved to {OUTPUT_DIR}")
