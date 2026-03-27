import cv2

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Invalid image path: {path}")
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)