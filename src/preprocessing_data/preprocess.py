"""
===========================================================
MODULE: Glyph Preprocessing
===========================================================

This module provides functions for cleaning, denoising,
and normalizing glyph images extracted from OCR/segmentation.

Pipeline:
    Input Image → Noise Removal → Smoothing → Normalization

"""

import cv2
import numpy as np



def preprocess_image(img):
    """
    Convert raw grayscale glyph image into a clean binary representation.

    Steps:
        1. Noise reduction using Gaussian blur
        2. Binarization using Otsu thresholding
        3. Standardize foreground polarity (black glyphs on white background)
        4. Morphological closing to repair broken strokes

    Args:
        img (np.ndarray): Input grayscale image

    Returns:
        np.ndarray: Clean binary image

    Raises:
        ValueError: If input image is None
    """

    if img is None:
        raise ValueError("input_image is None")



    # Blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Otsu Threshold
    _, img = cv2.threshold(img, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure BLACK glyph
    if np.mean(img) < 127:
        img = cv2.bitwise_not(img)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Stroke thickening if needed
    # img_inv = 255 - img
    # img_inv = cv2.dilate(img_inv, kernel, iterations=1)
    # img = 255 - img_inv

    # Closing
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, 1)

    return img




def process_glyph_image(input_image, filename="glyph.png"):
    """
    Normalize a preprocessed glyph into a fixed canvas.

    This function:
        - Extracts bounding box
        - Rescales glyph based on type (capital/small/descender)
        - Places glyph on standardized canvas

    Args:
        input_image (np.ndarray): Raw grayscale glyph image
        filename (str): Used for glyph classification

    Returns:
        np.ndarray or None: Normalized glyph image (120x140)
    """

    if input_image is None:
        raise ValueError("input_image is None")

    RED, BLUE, GREEN, YELLOW = 15, 45, 75, 105

    def get_bbox(img):
        """
        Compute bounding box of foreground pixels.

        Args:
            img (np.ndarray): Binary image

        Returns:
            tuple or None: (y0, y1, x0, x1)
        """
        coords = np.column_stack(np.where(img < 250))
        if coords.size == 0:
            return None
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return y0, y1, x0, x1

    def classify(name):
        """
        Heuristic classification based on filename.

        Categories:
            1 → capital letters / digits
            2 → normal lowercase glyphs
            3 → descenders (g, p, q, y, j)

        Args:
            name (str): filename

        Returns:
            int: class label (1, 2, or 3)
        """
        name = name.lower()
        if name.startswith("capital") or name[0].isdigit():
            return 1
        if any(k in name for k in ["small_g", "small_p", "small_q", "small_y", "small_j"]):
            return 3
        return 2

    binary = preprocess_image(input_image)

    bbox = get_bbox(binary)
    if bbox is None:
        return None

    y0, y1, x0, x1 = bbox
    glyph = binary[y0:y1+1, x0:x1+1]

    h, w = glyph.shape
    gtype = classify(filename)

    if gtype == 1:
        top, bottom = RED, GREEN
    elif gtype == 2:
        top, bottom = BLUE, GREEN
    else:
        top, bottom = BLUE, YELLOW

    target_h = bottom - top

    scale = target_h / h
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    glyph = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.ones((120, 140), dtype=np.uint8) * 255

    x_offset = (140 - new_w) // 2
    y_offset = top

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = glyph

    return canvas