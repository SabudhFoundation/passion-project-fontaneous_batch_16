import cv2
import numpy as np
from .config import *

def compute_char_score(img, ocr_conf):

    ink = (img < 128).astype(np.uint8)
    h, w = ink.shape

    # Fill ratio
    fill_ratio = np.sum(ink) / (h * w)
    fill_score = 1.0 - abs(fill_ratio - 0.18)

    # Connected components
    nlabels, _, _, _ = cv2.connectedComponentsWithStats(ink, 8)
    cc_score = 1.0 if nlabels <= 3 else 0.5

    # Aspect ratio
    coords = cv2.findNonZero(ink)
    if coords is None:
        return 0

    x, y, bw, bh = cv2.boundingRect(coords)
    aspect = bw / (bh + 1e-5)
    aspect_score = 1.0 - abs(aspect - 0.5)

    # Centering
    cx = x + bw // 2
    cy = y + bh // 2
    center_dist = np.sqrt((cx - w//2)**2 + (cy - h//2)**2)
    center_score = 1.0 - (center_dist / (w//2))

    score = (
        W_OCR * ocr_conf +
        W_FILL * fill_score +
        W_CC * cc_score +
        W_ASPECT * aspect_score +
        W_CENTER * center_score
    )

    return score