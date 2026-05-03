from .ocr_engine import OCREngine
from .normalization import normalize_char_image
from .scoring import compute_char_score
from .utils import get_label_folder

class Labeler:

    def __init__(self):
        self.ocr = OCREngine()

    def label_segments(self, segments):

        labeled = []

        for img in segments:

            result = self.ocr.get_single_char(img)

            if result is None:
                continue

            label, conf = result

            normalized = normalize_char_image(img, label)

            if normalized is None:
                continue

            score = compute_char_score(normalized, conf)

            labeled.append({
                "img": normalized,
                "label": get_label_folder(label),
                "conf": conf,
                "score": score
            })

        return labeled