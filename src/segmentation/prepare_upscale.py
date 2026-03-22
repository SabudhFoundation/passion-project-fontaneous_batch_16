import os
import cv2

# ── Tunable params ────────────────────────────────────────────
SCALE_F       = 10      # upscale factor for Tesseract
# ─────────────────────────────────────────────────────────────
class Prepare_upscale:
    def __init__(self, IMAGE_PATH, SCALE_F):
        self.img_bgr = cv2.imread(IMAGE_PATH)
        self.SCALE_F=SCALE_F


    def run_prep(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W    = gray.shape
        return img_rgb, gray, H, W

    def clache(self,gray):
        # CLAHE enhancement for OCR
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        scaled = cv2.resize(enhanced, None, fx=self.SCALE_F, fy=self.SCALE_F,
                    interpolation=cv2.INTER_CUBIC)
        return  enhanced,scaled
    

