from .ocr_engine import OCREngine
from .normalization import normalize_char_image
from .scoring import compute_char_score

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

            norm = normalize_char_image(img)

            if norm is None:
                continue

            score = compute_char_score(norm, conf)

            labeled.append({
                "img": norm,
                "label": label,
                "conf": conf,
                "score": score
            })

        return labeled

    def group_by_label(self, labeled_data):

        groups = {}

        for item in labeled_data:
            label = item["label"]

            if label not in groups:
                groups[label] = []

            groups[label].append(item)

        # sort by score
        for label in groups:
            groups[label].sort(key=lambda x: x["score"], reverse=True)

        return groups