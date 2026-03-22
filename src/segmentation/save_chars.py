import json
from PIL import Image
import os

class CharacterCropper:
    def __init__(self, output_dir="output", score_threshold=0.05, padding=4):
        self.output_dir      = output_dir
        self.score_threshold = score_threshold
        self.padding         = padding
        os.makedirs(self.output_dir, exist_ok=True)

    def crop_and_save(self, matched_chars, img_rgb):
        H, W = img_rgb.shape[:2]
        saved, skipped, metadata = [], [], []

        sorted_chars = sorted(matched_chars, key=lambda x: (x[1][1], x[1][0]))

        for i, (ch, (bx0, by0, bx1, by1), score, ocr_box) in enumerate(sorted_chars):
            if score < self.score_threshold:
                skipped.append((ch, score))
                continue

            px0 = max(0, bx0 - self.padding);  px1 = min(W, bx1 + self.padding)
            py0 = max(0, by0 - self.padding);  py1 = min(H, by1 + self.padding)
            crop = img_rgb[py0:py1, px0:px1]

            safe_ch = ch if ch.isalnum() else f"ord{ord(ch)}"
            fname   = f"char_{i:03d}_{safe_ch}_score{score:.2f}.png"
            fpath   = os.path.join(self.output_dir, fname)
            Image.fromarray(crop).save(fpath)

            saved.append((ch, fname, score))
            metadata.append({
                "index": i, "char": ch, "file": fname,
                "matched_bbox": [int(bx0), int(by0), int(bx1), int(by1)],
                "ocr_bbox":     [int(b) for b in ocr_box],
                "score":        round(float(score), 4)
            })

        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return saved, skipped