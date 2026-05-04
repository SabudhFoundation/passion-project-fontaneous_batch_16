"""
Fontaneous Pipeline

USAGE:
    python src/main.py --dataset dataset

INPUT:
    dataset/
        kid1/*.jpg
        kid2/*.jpg

OUTPUT (inside src/data/):
    final_dataset/<kid>/<label>/images/*.png
    best_chars/<kid>/*.png
    best_chars_inverted/<kid>/*.png
    output_svgs/<kid>/*.svg
    font_file/<kid>.ttf

Fonts are also copied to:
    ~/Downloads/

NOTE:
    • One font generated per sub-folder (kid1.ttf, kid2.ttf)
    • CLI selection step requires manual input
"""


import os
import cv2
import argparse
from pathlib import Path

import preprocessing_data.preprocessing as prep
from segmentation.Segmentation import process_segemntation
from char_ocr_labeling.main import run_ocr_pipeline
from char_ocr_labeling.cli_selection import interactive_best_selection_cli
from char_ocr_labeling.saver import save_final_outputs

from vectorization.service import generate_svgs
from ttf_generation.ttf_final import build_font

from utils.logger import log_step, log_info, log_done
from utils.io import copy_to_downloads


# =========================================
# PATH CONFIG
# =========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

SVG_ROOT = DATA_ROOT / "output_svgs"
FONT_DIR = DATA_ROOT / "font_file"


# =========================================
# ARGUMENTS
# =========================================
parser = argparse.ArgumentParser(description="Dataset → Multi TTF Pipeline")
parser.add_argument("--dataset", required=True, help="Dataset folder path")


# =========================================
# RUN
# =========================================
if __name__ == "__main__":

    args = parser.parse_args()
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Iterate over each kid folder
    for kid_folder in sorted(dataset_path.iterdir()):

        if not kid_folder.is_dir():
            continue

        font_name = kid_folder.name
        ttf_output = FONT_DIR / f"{font_name}.ttf"

        log_step(f"Processing {font_name}")

        all_crops = []

        # -----------------------------------------
        # READ ALL IMAGES
        # -----------------------------------------
        for img_path in kid_folder.glob("*.*"):

            if not img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                continue

            log_info(f"Reading {img_path.name}")

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            pre = prep.preprocess_single(img)
            crops = process_segemntation(pre)

            all_crops.extend(crops)

        if not all_crops:
            log_info("No characters found, skipping")
            continue

        log_info(f"Total chars: {len(all_crops)}")

        # -----------------------------------------
        # OCR
        # -----------------------------------------
        log_step("OCR + Labeling")
        ocr = run_ocr_pipeline(all_crops)

        if not ocr["labeled"]:
            log_info("No valid OCR, skipping")
            continue

        # -----------------------------------------
        # SELECTION
        # -----------------------------------------
        data = {font_name: {"labeled": ocr["labeled"]}}

        log_step("Manual Selection")
        data = interactive_best_selection_cli(data)

        # -----------------------------------------
        # SAVE DATASET
        # -----------------------------------------
        log_step("Saving dataset")
        save_final_outputs(data)

        # -----------------------------------------
        # SVG
        # -----------------------------------------
        log_step("Vectorization")
        svg_folder = generate_svgs(data, font_name, SVG_ROOT)

        # -----------------------------------------
        # TTF
        # -----------------------------------------
        log_step("Building TTF")

        build_font(
            base_folder=str(SVG_ROOT),
            folders=[font_name],
            output_ttf=str(ttf_output),
            font_name=font_name,
            delete_svgs=False,
        )

        log_done(f"{font_name}.ttf created")

        # -----------------------------------------
        # COPY TO DOWNLOADS
        # -----------------------------------------
        copy_to_downloads(ttf_output)

    print("\nAll fonts generated successfully!")