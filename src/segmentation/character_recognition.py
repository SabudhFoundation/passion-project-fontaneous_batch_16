import cv2
from skimage.metrics import structural_similarity as ssim


class character_recognition:
    def __init__(self, PATCH_SIZE,MIN_SSIM_WIN,patch_scores,attentions,enhanced,ocr_chars,H,W,ph,pw):
        self.patch_scores = patch_scores
        self.attentions = attentions
        self.enhanced = enhanced
        self.W = W
        self.H = H
        self.ocr_chars = ocr_chars
        self.PATCH_SIZE=PATCH_SIZE
        self.MIN_SSIM_WIN = MIN_SSIM_WIN
        self.ph=ph
        self.pw=pw
    
    def get_attention_score(self,x0, y0, x1, y1):
        """Mean ViT patch attention over a bounding box region."""
        px0 = max(0, x0 // self.PATCH_SIZE)
        py0 = max(0, y0 // self.PATCH_SIZE)
        px1 = min(self.pw, (x1 + self.PATCH_SIZE - 1) // self.PATCH_SIZE)
        py1 = min(self.ph, (y1 + self.PATCH_SIZE - 1) // self.PATCH_SIZE)
        if px1 <= px0 or py1 <= py0:
            return 0.0
        return float(self.patch_scores[py0:py1, px0:px1].mean())

    def get_ssim_score(self,gray_img, x0, y0, x1, y1):
        """SSIM-based texture complexity score for a region."""
        roi = gray_img[y0:y1, x0:x1]
        if roi.size == 0 or roi.shape[0] < self.MIN_SSIM_WIN or roi.shape[1] < self.MIN_SSIM_WIN:
            return 0.0
        win = min(self.MIN_SSIM_WIN, roi.shape[0]-1, roi.shape[1]-1)
        if win < 3:
            return float(roi.std()) / 128.0
        # Compare ROI against its own blurred version — measures local structure
        blurred = cv2.GaussianBlur(roi, (win|1, win|1), 0)
        score, _ = ssim(roi, blurred, full=True, data_range=255)
        # Invert: high SSIM to blur = smooth/empty; we want complex/inky
        return float(1.0 - max(0, score))

    def find_best_char_region(self, gray_img, char_w, char_h, sx0, sy0, sx1, sy1):
        """Slide a char-sized window over search area, return best-scoring position."""
        best_score = -1
        best_box   = (sx0, sy0, sx0+char_w, sy0+char_h)
        step = max(2, min(char_w, char_h) // 4)
        for y in range(sy0, max(sy0+1, sy1-char_h), step):
            for x in range(sx0, max(sx0+1, sx1-char_w), step):
                x1c, y1c = min(x+char_w, self.W), min(y+char_h, self.H)
                attn  = self.get_attention_score(x, y, x1c, y1c)
                tex   = self.get_ssim_score(gray_img, x, y, x1c, y1c)
                score = 0.6*attn + 0.4*tex
                if score > best_score:
                    best_score = score
                    best_box   = (x, y, x1c, y1c)
        return best_box, best_score

    def match_characters(self, SEARCH_PAD):
        """Run matching for every OCR character."""
        print("Matching characters...\n")
        print(f"{'#':>3}  {'Chr':^5}  {'OCR Box':^24}  {'Best Box':^24}  {'Score':>7}")
        print("─" * 75)

        matched_chars = []
        for i, (ch, ox0, oy0, ox1, oy1) in enumerate(self.ocr_chars):
            cw   = max(ox1-ox0, 8)
            ch_h = max(oy1-oy0, 8)

            sx0 = max(0, ox0 - SEARCH_PAD)
            sy0 = max(0, oy0 - SEARCH_PAD)
            sx1 = min(self.W, ox1 + SEARCH_PAD)
            sy1 = min(self.H, oy1 + SEARCH_PAD)

            best_box, score = self.find_best_char_region(self.enhanced, cw, ch_h, sx0, sy0, sx1, sy1)
            matched_chars.append((ch, best_box, score, (ox0, oy0, ox1, oy1)))

            bx0, by0, bx1, by1 = best_box
            print(f"{i:>3}  {repr(ch):^5}  ({ox0:3},{oy0:3},{ox1:3},{oy1:3})  "
                  f"({bx0:3},{by0:3},{bx1:3},{by1:3})  {score:7.4f}")

        print(f"\n {len(matched_chars)} characters matched")
        return matched_chars
