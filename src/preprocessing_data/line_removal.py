import cv2
import numpy as np

def inpaint_horizontal_lines(gray):
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (101, 101))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, se)
    diff = cv2.divide(gray, bg, scale=255)

    line_thresh = cv2.adaptiveThreshold(
        diff, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )

    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    lines = cv2.morphologyEx(line_thresh, cv2.MORPH_OPEN, horiz_k)

    mask = cv2.dilate(lines, np.ones((3,1), np.uint8))

    healed = cv2.inpaint(diff, mask, 1, cv2.INPAINT_TELEA)
    return healed