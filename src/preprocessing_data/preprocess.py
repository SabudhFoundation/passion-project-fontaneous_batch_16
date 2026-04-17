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
    Preprocess a grayscale glyph image into a clean binary format.

    Steps:
        1. Apply Gaussian blur to reduce noise
        2. Perform Otsu thresholding for binarization
        3. Ensure consistent polarity (black glyph on white background)
        4. Apply morphological closing to fix broken strokes

    Args:
        img (np.ndarray): Input grayscale image

    Returns:
        np.ndarray: Processed binary image

    Raises:
        ValueError: If input image is None
        RuntimeError: If any processing step fails
    """
    try:
        # Validate input
        if img is None:
            raise ValueError("input_image is None")

        # Step 1: Noise reduction using Gaussian Blur
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Step 2: Binarization using Otsu Thresholding
        _, img = cv2.threshold(
            img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Step 3: Ensure glyph is black (foreground)
        if np.mean(img) < 127:
            img = cv2.bitwise_not(img)

        # Step 4: Morphological closing to repair strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, 1)

        return img

    except Exception as e:
        raise RuntimeError(f"Error in preprocess_image: {e}")


def process_glyph_image(input_image, filename="glyph.png"):
    """
    Normalize a glyph image into a fixed-size standardized canvas.

    This function:
        - Extracts the bounding box of the glyph
        - Classifies glyph type based on filename
        - Rescales glyph according to type (capital/small/descender)
        - Places glyph on a fixed canvas (120x140)

    Args:
        input_image (np.ndarray): Raw grayscale glyph image
        filename (str): Filename used for heuristic classification

    Returns:
        np.ndarray or None:
            Normalized glyph image (120x140) or None if no glyph found

    Raises:
        ValueError: If input image is None
        RuntimeError: If processing fails
    """
    try:
        # Validate input
        if input_image is None:
            raise ValueError("input_image is None")

        # Predefined vertical alignment zones
        RED, BLUE, GREEN, YELLOW = 15, 45, 75, 105

        def get_bbox(img):
            """
            Compute bounding box of foreground pixels.

            Args:
                img (np.ndarray): Binary image

            Returns:
                tuple or None: (y0, y1, x0, x1) or None if empty
            """
            coords = np.column_stack(np.where(img < 250))
            if coords.size == 0:
                return None

            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            return y0, y1, x0, x1

        def classify(name):
            """
            Classify glyph type based on filename.

            Categories:
                1 → Capital letters / digits
                2 → Normal lowercase letters
                3 → Descenders (g, p, q, y, j)

            Args:
                name (str): Filename

            Returns:
                int: Class label (1, 2, or 3)
            """
            name = name.lower()

            if name.startswith("capital") or name[0].isdigit():
                return 1

            if any(k in name for k in
                   ["small_g", "small_p", "small_q", "small_y", "small_j"]):
                return 3

            return 2

        # Step 1: Preprocess image
        binary = preprocess_image(input_image)

        # Step 2: Extract bounding box
        bbox = get_bbox(binary)
        if bbox is None:
            return None  # No glyph detected

        y0, y1, x0, x1 = bbox
        glyph = binary[y0:y1 + 1, x0:x1 + 1]

        # Step 3: Get glyph dimensions and type
        h, w = glyph.shape
        gtype = classify(filename)

        # Step 4: Define vertical placement based on glyph type
        if gtype == 1:
            top, bottom = RED, GREEN
        elif gtype == 2:
            top, bottom = BLUE, GREEN
        else:
            top, bottom = BLUE, YELLOW

        # Step 5: Compute scaling factor
        target_h = bottom - top
        scale = target_h / h

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # Step 6: Resize glyph
        glyph = cv2.resize(
            glyph,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )

        # Step 7: Create blank canvas
        canvas = np.ones((120, 140), dtype=np.uint8) * 255

        # Step 8: Center horizontally and place vertically
        x_offset = (140 - new_w) // 2
        y_offset = top

        canvas[y_offset:y_offset + new_h,
               x_offset:x_offset + new_w] = glyph

        return canvas

    except Exception as e:
        raise RuntimeError(
            f"Error in process_glyph_image ({filename}): {e}"
        )