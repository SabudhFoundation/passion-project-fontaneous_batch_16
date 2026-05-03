"""
Dataset saving utilities for OCR pipeline.

This module handles:
    - Saving full labeled dataset (organized by label)
    - Saving best-selected characters per input
    - Generating inverted variants for training diversity

Output structure:
    final_dataset/<sub>/<label>/images/*.png
    best_chars/<sub>/*.png
    best_chars_inverted/<sub>/*.png
"""

import os
import cv2
import numpy as np


# =========================================
# OUTPUT ROOTS
# =========================================
DATASET_ROOT = "final_dataset"
BEST_ROOT = "best_chars"
BEST_INV_ROOT = "best_chars_inverted"


# =========================================
# SAFE WRITE
# =========================================
def safe_write(path, img):
    """
    Safely write an image to disk.

    Args:
        path (str): Output file path
        img (np.ndarray): Image to save

    Notes:
        - Uses cv2.imwrite
        - Logs a warning if writing fails (no exception raised)
    """
    ok = cv2.imwrite(path, img)
    if not ok:
        print(f"Failed to save: {path}")


# =========================================
# MAIN SAVE FUNCTION
# =========================================
def save_final_outputs(data):
    """
    Save labeled OCR dataset and selected best characters to disk.

    This function performs two main operations per input group:
        1. Save all labeled samples into structured dataset folders
        2. Save best-selected characters (normal + inverted)

    Args:
        data (dict):
            {
                <sub_name>: {
                    "labeled": [
                        {
                            "img": np.ndarray,
                            "label": str,
                            "conf": float,
                            "score": float
                        }
                    ],
                    "best": [
                        {
                            "img": np.ndarray,
                            "label": str,
                            "conf": float,
                            "score": float
                        }
                    ]
                }
            }

    Output:
        - Full dataset:
            final_dataset/<sub>/<label>/images/<idx>.png

        - Best samples:
            best_chars/<sub>/<label>.png
            best_chars_inverted/<sub>/<label>.png

    Behavior:
        - Creates directories if they do not exist
        - Prevents overwrite using random suffix for duplicate filenames
        - Skips best-saving step if no selections exist

    Assumptions:
        - Images are preprocessed and normalized (e.g., 64x64 binary format)
        - Labels are already mapped to folder-safe names

    Notes:
        - Inverted images are useful for model robustness
        - No exceptions are raised for write failures (logged instead)
    """

    # Ensure root directories exist
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(BEST_ROOT, exist_ok=True)
    os.makedirs(BEST_INV_ROOT, exist_ok=True)

    for sub in data:

        print(f"\nSaving: {sub}")

        best_sub_dir = os.path.join(BEST_ROOT, sub)
        best_inv_sub_dir = os.path.join(BEST_INV_ROOT, sub)

        os.makedirs(best_sub_dir, exist_ok=True)
        os.makedirs(best_inv_sub_dir, exist_ok=True)

        # =========================================
        # 1. SAVE FULL DATASET
        # =========================================
        for idx, item in enumerate(data[sub]["labeled"]):

            img = item["img"]
            label_name = item["label"]

            label_dir = os.path.join(
                DATASET_ROOT,
                sub,
                label_name,
                "images"
            )
            os.makedirs(label_dir, exist_ok=True)

            fname = f"{idx}.png"
            path = os.path.join(label_dir, fname)

            safe_write(path, img)

        # =========================================
        # 2. SAVE BEST SAMPLES
        # =========================================
        if "best" not in data[sub] or len(data[sub]["best"]) == 0:
            print(f"No best selected for {sub}")
            continue

        for item in data[sub]["best"]:

            label_name = item["label"]
            norm_img = item["img"]

            fname = f"{label_name}.png"

            normal_path = os.path.join(best_sub_dir, fname)
            inv_path = os.path.join(best_inv_sub_dir, fname)

            # Prevent overwrite by appending random suffix
            if os.path.exists(normal_path):
                fname = f"{label_name}_{np.random.randint(1e6)}.png"
                normal_path = os.path.join(best_sub_dir, fname)
                inv_path = os.path.join(best_inv_sub_dir, fname)

            # Save normal image
            safe_write(normal_path, norm_img)

            # Save inverted version
            inv_img = cv2.bitwise_not(norm_img)
            safe_write(inv_path, inv_img)

            print(f"Saved: {sub}/{fname}")

    print("\nSaving complete")