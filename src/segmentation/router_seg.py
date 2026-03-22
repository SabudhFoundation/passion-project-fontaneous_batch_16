import os
import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from prepare_upscale       import Prepare_upscale
from ocr_boxes             import Ocr_boxes
from attention_mapping     import attention_mapping
from character_recognition import character_recognition
from save_chars            import CharacterCropper

router = APIRouter(prefix="/ocr", tags=["OCR"])

SCORE_THRESHOLD = 0.05
PATCH_SIZE      = 16
SCALE_F         = 10
SEARCH_PAD      = 50
MIN_SSIM_WIN    = 16
PADDING         = 8

MODEL_NAME    = "google/vit-base-patch16-224"
DISCARD_RATIO = 0.9
HEAD_FUSION   = "mean"
TARGET_SIZE   = 224


class RunRequest(BaseModel):
    output_dir: str = "output"


@router.post("/segmentation")
def run_pipeline(image_path: str, req: RunRequest):
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # 1. Prepare
    prep                    = Prepare_upscale(image_path, SCALE_F)
    img_rgb, gray, H, W     = prep.run_prep(prep.img_bgr)
    enhanced, scaled        = prep.clache(gray)

    # 2. OCR
    ocr_chars = Ocr_boxes(img_rgb, scaled, SCALE_F).bounding_boxes()

    # 3. ViT attention
    patch_scores, attentions, pw, ph = attention_mapping(
        img_rgb, MODEL_NAME, DISCARD_RATIO, HEAD_FUSION, TARGET_SIZE, PATCH_SIZE
    ).load_vit_model()

    # 4. Match characters
    matched_chars = character_recognition(
        PATCH_SIZE, MIN_SSIM_WIN, patch_scores, attentions,
        enhanced, ocr_chars, H, W, ph, pw
    ).match_characters(SEARCH_PAD)

    # 5. Save
    saved, skipped = CharacterCropper(req.output_dir, SCORE_THRESHOLD, PADDING)\
        .crop_and_save(matched_chars, img_rgb)

    return {"saved": len(saved), "skipped": len(skipped), "output_dir": req.output_dir}