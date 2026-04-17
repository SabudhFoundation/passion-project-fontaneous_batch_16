import os
import cv2
import numpy as np

DATASET_ROOT = "final_dataset"
BEST_ROOT = "best_chars"
BEST_INV_ROOT = "best_chars_inverted"


def safe_write(path, img):
    ok = cv2.imwrite(path, img)
    if not ok:
        print(f"Failed to save: {path}")


def save_final_outputs(data):

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
        # 1. SAVE FULL DATASET (KID → LABEL → images/)
        # =========================================
        for idx, item in enumerate(data[sub]["labeled"]):

            img = item["img"]
            raw_label = item["label"]

            label_name = raw_label

            # REQUIRED STRUCTURE
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
        # 2. SAVE BEST (FLAT PER KID)
        # =========================================
        if "best" not in data[sub] or len(data[sub]["best"]) == 0:
            print(f"No best selected for {sub}")
            continue

        for item in data[sub]["best"]:

            raw_label = item["label"]
            norm_img = item["img"]

            label_name = raw_label

            fname = f"{label_name}.png"

            normal_path = os.path.join(best_sub_dir, fname)
            inv_path = os.path.join(best_inv_sub_dir, fname)

            # prevent overwrite
            if os.path.exists(normal_path):
                fname = f"{label_name}_{np.random.randint(1e6)}.png"
                normal_path = os.path.join(best_sub_dir, fname)
                inv_path = os.path.join(best_inv_sub_dir, fname)

            # save normal
            safe_write(normal_path, norm_img)

            # save inverted
            inv_img = cv2.bitwise_not(norm_img)
            safe_write(inv_path, inv_img)

            print(f"Saved: {sub}/{fname}")

    print("\nSaving complete")