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


IMG_SIZE = 64


def preprocess_image(img):
    """
    Perform full preprocessing on a glyph image.

    Steps:
        1. Resize to fixed size
        2. Apply Gaussian blur
        3. Apply Otsu thresholding
        4. Ensure black foreground
        5. Stroke thickening
        6. Morphological closing
        7. Remove far components (based on centroid distance)
        8. Remove small noise (area filtering)
        9. Smooth contours
        10. Final thresholding

    Args:
        img (np.ndarray): Input grayscale image

    Returns:
        np.ndarray: Cleaned binary image
    """

    if img is None:
        raise ValueError("input_image is None")

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Otsu Threshold
    _, img = cv2.threshold(img, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure BLACK glyph
    if np.mean(img) < 127:
        img = cv2.bitwise_not(img)

    # Stroke thickening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_inv = 255 - img
    img_inv = cv2.dilate(img_inv, kernel, iterations=1)
    img = 255 - img_inv

    # Closing
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, 1)

    # -------------------------------
    # Remove FAR components
    # -------------------------------
    inv = cv2.bitwise_not(img)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, 8)

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        main_centroid = centroids[largest_label]

        DIST_THRESHOLD = 20
        filtered = np.zeros_like(inv)

        for i in range(1, num_labels):
            cx, cy = centroids[i]
            dist = np.linalg.norm(np.array([cx, cy]) - np.array(main_centroid))

            if dist < DIST_THRESHOLD:
                filtered[labels == i] = 255

        img = cv2.bitwise_not(filtered)

    # -------------------------------
    # Area-based noise removal
    # -------------------------------
    MIN_NOISE_AREA = int(IMG_SIZE * IMG_SIZE * 0.005)

    inv = cv2.bitwise_not(img)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)

    cleaned = np.zeros_like(inv)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_NOISE_AREA:
            cleaned[labels == i] = 255

    img = cv2.bitwise_not(cleaned)

    # -------------------------------
    # Contour smoothing
    # -------------------------------
    contours, hierarchy = cv2.findContours(
        255 - img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    smooth = np.ones_like(img) * 255

    if hierarchy is not None:
        hierarchy = hierarchy[0]

        for i, cnt in enumerate(contours):
            epsilon = 0.009 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if hierarchy[i][3] == -1:
                cv2.drawContours(smooth, [approx], -1, 0, -1)
            else:
                cv2.drawContours(smooth, [approx], -1, 255, -1)

    img = smooth

    # Final smoothing
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

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
        coords = np.column_stack(np.where(img < 250))
        if coords.size == 0:
            return None
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return y0, y1, x0, x1

    def classify(name):
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