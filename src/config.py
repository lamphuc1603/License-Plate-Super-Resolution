"""
Configuration loader for the ICIP-XLPSR inference pipeline.

Static settings live in config.yaml; this module reads them and derives
the runtime objects used across the pipeline (device, compiled regexes,
lookup tables, resolved paths).
"""

import re
from pathlib import Path

import yaml
import torch

# =====================================================================
# Load static settings from config.yaml (next to this file)
# =====================================================================
_SRC_DIR  = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent  # project root (folder containing src/)

_CFG_PATH = _SRC_DIR / "config.yaml"
with open(_CFG_PATH, "r", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)


def _resolve(p):
    """Absolute paths are kept as-is; relative ones resolve against the root."""
    p = Path(p)
    return p if p.is_absolute() else (_ROOT_DIR / p)

# =====================================================================
# Device
# =====================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# Paths
# =====================================================================
_paths = _CFG["paths"]
INPUT_DIR  = _resolve(_paths["input_dir"])
OUTPUT_DIR = _resolve(_paths["output_dir"])

CKPT_DIR   = _resolve(_paths["ckpt_dir"])
TPGSR_CKPT = CKPT_DIR / _paths["tpgsr_ckpt"]
CRNN_CKPT  = CKPT_DIR / _paths["crnn_ckpt"]
TROCR_DIRS = [CKPT_DIR / f"trocr_{s}" for s in _paths["trocr_subdirs"]]

# =====================================================================
# Plate constants
# =====================================================================
_plate = _CFG["plate"]
CHARACTERS  = _plate["characters"]
CHAR_DICT   = {c: i for i, c in enumerate(CHARACTERS)}
NUM_CLASSES = _plate["num_classes"]
BLANK_IDX   = _plate["blank_idx"]

PLATE_REGEXES  = {int(k): re.compile(v) for k, v in _plate["regexes"].items()}
PLATE_PATTERNS = {int(k): v for k, v in _plate["patterns"].items()}
PLATE_LENGTHS  = {int(k): v for k, v in _plate["lengths"].items()}
VALID_LETTERS  = set(c for c in CHARACTERS if c.isalpha())

# =====================================================================
# Image dimensions
# =====================================================================
_img = _CFG["image"]
LR_H, LR_W = _img["lr_h"], _img["lr_w"]
HR_H, HR_W = _img["hr_h"], _img["hr_w"]

# =====================================================================
# Inference hyperparameters
# =====================================================================
_inf = _CFG["inference"]
MAX_TARGET_LEN    = _inf["max_target_len"]
NUM_BEAMS         = _inf["num_beams"]
TOP_K             = _inf["top_k"]
UNDERSCORE_THRESH = _inf["underscore_thresh"]
INVALID_THRESH    = _inf["invalid_thresh"]
OCR1_WEIGHT       = _inf["ocr1_weight"]
FILL_THRESH       = _inf["fill_thresh"]
