"""Glyph preprocessing for vectorization."""

import cv2
import numpy as np


CANVAS_W = 500
CANVAS_H = 700

RED = 50
BLUE = 200
GREEN = 500
YELLOW = 620


def preprocess_image(img):
    """Return a clean glyph image with black ink on white background."""
    try:
        if img is None:
            raise ValueError("input_image is None")

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.GaussianBlur(img, (5, 5), 0)

        _, img = cv2.threshold(
            img,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        if np.mean(img) < 127:
            img = cv2.bitwise_not(img) ## inverting colors if background is darker

<<<<<<< HEAD
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, open_kernel, iterations=1)
=======
        # Step 4: Morphological closing to repair strokes(fill gaps)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, 1)
>>>>>>> 46c7ba66b2c09f31aa3f537b61ff10762cdae14d

        return img

    except Exception as e:
        raise RuntimeError(f"Error in preprocess_image: {e}")


def process_glyph_image(input_image, filename="glyph.png"):
    """Normalize a glyph image on a fixed canvas for vectorization."""
    try:
        if input_image is None:
            raise ValueError("input_image is None")

        def get_bbox(img):
            coords = np.column_stack(np.where(img < 245))
            if coords.size == 0:
                return None

            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            return y0, y1, x0, x1

        # Classification
        def classify(name):
<<<<<<< HEAD
=======
            """
            Classify the glyph type based on its filename.

            The classification determines how the glyph will be vertically
            aligned and scaled within typographic zones.

            Categories:
                1 → Capital letters or digits
                2 → Normal lowercase letters
                3 → Ascender letters (e.g., b, d, h, l)
                4 → Descender letters (e.g., g, p, q, y, j)

            Parameters:
                name (str): Filename or label of the glyph.

            Returns:
                int: Glyph type category (1 to 4).
            """
>>>>>>> 46c7ba66b2c09f31aa3f537b61ff10762cdae14d
            name = name.lower()

            # Capital
            if name.startswith("capital") or name[0].isdigit():
                return 1

            # Descenders
            if any(k in name for k in
                   ["small_g", "small_p", "small_q", "small_y", "small_j"]):
                return 4

<<<<<<< HEAD
            if any(k in name for k in
                   ["small_b", "small_d", "small_f", "small_h",
                    "small_k", "small_l", "small_t"]):
=======
            # Ascenders
            if any(k in name for k in
                   ["small_b", "small_d", "small_f", "small_h", "small_k", "small_l"]):
>>>>>>> 46c7ba66b2c09f31aa3f537b61ff10762cdae14d
                return 3

            # Normal lowercase
            return 2
<<<<<<< HEAD

        binary = preprocess_image(input_image)

=======
        
        # Step 1: Preprocess
        binary = preprocess_image(input_image)

        # Step 1: Bounding box
>>>>>>> 46c7ba66b2c09f31aa3f537b61ff10762cdae14d
        bbox = get_bbox(binary)
        if bbox is None:
            return None

        y0, y1, x0, x1 = bbox
        glyph = binary[y0:y1 + 1, x0:x1 + 1]
        glyph = cv2.copyMakeBorder(
            glyph,
            2,
            2,
            2,
            2,
            cv2.BORDER_CONSTANT,
            value=255,
        )

<<<<<<< HEAD
        h, w = glyph.shape
        gtype = classify(filename)

        if gtype == 1:
=======
        # Step 2
        h, w = glyph.shape
        gtype = classify(filename)

        # Zones
        if gtype == 1:  # capital
>>>>>>> 46c7ba66b2c09f31aa3f537b61ff10762cdae14d
            top, bottom = RED, GREEN
        elif gtype == 2:  # normal
            top, bottom = BLUE, GREEN
<<<<<<< HEAD
        elif gtype == 3:
            top, bottom = RED, GREEN
        else:
            top, bottom = BLUE, YELLOW

=======
        elif gtype == 3:  # ascender
            top, bottom = RED, GREEN
        else:  # descender
            top, bottom = BLUE, YELLOW

        # Step 3: Scaling
>>>>>>> 46c7ba66b2c09f31aa3f537b61ff10762cdae14d
        target_h = bottom - top
        scale = target_h / h
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))

<<<<<<< HEAD
        if new_w > CANVAS_W - 40:
            new_w = CANVAS_W - 40
            new_h = max(1, int(h * (new_w / w)))

        interpolation = cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA
        glyph = cv2.resize(glyph, (new_w, new_h), interpolation=interpolation)
        glyph = cv2.GaussianBlur(glyph, (3, 3), 0)

        canvas = np.ones((CANVAS_H, CANVAS_W), dtype=np.uint8) * 255
        x_offset = (CANVAS_W - new_w) // 2
=======
        # Step 4: Resize
        glyph = cv2.resize(
            glyph,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )

        # Step 5: Canvas
        canvas = np.ones((120, 140), dtype=np.uint8) * 255

        # Step 6: Placement
        x_offset = (140 - new_w) // 2
>>>>>>> 46c7ba66b2c09f31aa3f537b61ff10762cdae14d

        if "small_j" in filename.lower():
            y_offset = bottom - new_h
        else:
            y_offset = top
<<<<<<< HEAD

        y_offset = max(0, min(y_offset, CANVAS_H - new_h))
        x_offset = max(0, min(x_offset, CANVAS_W - new_w))
=======
>>>>>>> 46c7ba66b2c09f31aa3f537b61ff10762cdae14d

        canvas[y_offset:y_offset + new_h,
               x_offset:x_offset + new_w] = glyph

        return canvas

    except Exception as e:
        raise RuntimeError(f"Error in process_glyph_image ({filename}): {e}")
