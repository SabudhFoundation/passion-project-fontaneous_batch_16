import cv2
import numpy as np
from .config import IMG_SIZE, FILL_RATIO_TARGET

def normalize_char_image(img, size=IMG_SIZE):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)

    fg = (th == 0).astype(np.uint8)

    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)

    if nlabels <= 1:
        return None

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    clean_fg = np.zeros_like(fg)
    clean_fg[labels == largest_idx] = 1

    char_mask = (clean_fg * 255).astype(np.uint8)
    th = 255 - char_mask

    coords = cv2.findNonZero(char_mask)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    char = th[y:y+h, x:x+w]

    scale = min((size * FILL_RATIO_TARGET) / w, (size * FILL_RATIO_TARGET) / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    char_resized = cv2.resize(char, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.ones((size, size), dtype=np.uint8) * 255

    x1 = (size - new_w) // 2
    y1 = (size - new_h) // 2

    canvas[y1:y1+new_h, x1:x1+new_w] = char_resized

    return canvas