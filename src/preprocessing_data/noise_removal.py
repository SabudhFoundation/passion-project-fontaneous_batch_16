import cv2
import numpy as np

def remove_noise_floor(img, min_area=35):
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    cleaned = np.zeros_like(img)

    for i in range(1, nlabels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned