import easyocr
from config import OCR_CONF

class OCREngine:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def get_single_char(self, img):

        results = self.reader.readtext(img)

        best = None

        for _, text, conf in results:

            text = "".join(filter(str.isalnum, text))

            if len(text) != 1:
                continue

            if conf < OCR_CONF:
                continue

            if best is None or conf > best[1]:
                best = (text, conf)

        return best