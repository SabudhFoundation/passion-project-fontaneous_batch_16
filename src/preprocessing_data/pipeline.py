import cv2
from .image_loader import load_image, to_gray
from .line_removal import inpaint_horizontal_lines
from .noise_removal import remove_noise_floor
from .thresholding import adaptive_threshold

def process_image(path, output_path):
    img = load_image(path)
    gray = to_gray(img)

    clean = inpaint_horizontal_lines(gray)
    blurred = cv2.GaussianBlur(clean, (3,3), 0)

    binary = adaptive_threshold(blurred)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    final_bin = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    final = remove_noise_floor(final_bin)

    cv2.imwrite(output_path, final)