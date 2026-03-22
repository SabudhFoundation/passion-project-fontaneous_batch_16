
import pytesseract
import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"src/segmentation/Tesseract-OCR/tesseract.exe"

class Ocr_boxes:
        
    def __init__(self, img_rgb, scaled, SCALE_F):
        self.img_rgb = img_rgb
        self.scaled = scaled
        self.SCALE_F = SCALE_F
            

    def bounding_boxes(self):
        boxes_raw = pytesseract.image_to_boxes(self.scaled, config='--psm 6 --oem 1')
        sh = self.scaled.shape[0]

        ocr_chars = []
        for line in boxes_raw.strip().split('\n'):
            parts = line.split()
            if len(parts) < 5:
                continue
            ch, x1, y1, x2, y2 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            # Convert: tesseract uses bottom-left origin → flip y, scale down
            ox0 = x1 // self.SCALE_F
            oy0 = (sh - y2) // self.SCALE_F
            ox1 = x2 // self.SCALE_F
            oy1 = (sh - y1) // self.SCALE_F
            w, h = ox1 - ox0, oy1 - oy0
            if w > 3 and h > 3 and ch.strip():   # filter degenerate boxes
                ocr_chars.append((ch, ox0, oy0, ox1, oy1))

        print(f"Tesseract detected {len(ocr_chars)} characters:\n")
        return ocr_chars
