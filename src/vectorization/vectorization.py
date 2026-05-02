"""
===========================================================
MODULE: Image Vectorization
===========================================================

This module converts preprocessed glyph images into
vector graphics (SVG format) using VTracer.

Pipeline:
    Input Image → Vectorization → SVG Output

"""

import cv2
import vtracer
import tempfile
import os



def vectorize_image(input_path, output_path):
    """
    Convert raster image file to SVG using VTracer.

    Args:
        input_path (str): Path to input image
        output_path (str): Path to output SVG

    Returns:
        None
    """
    try:
        vtracer.convert_image_to_svg_py(
            input_path,
            output_path,
            colormode='binary',
            mode='spline',
            filter_speckle=6,
            corner_threshold=90,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=40
            )
    except Exception as e:
        print(f" VTracer error for {input_path}: {e}")


def process_vectorization(input_image):
    """
    Convert a preprocessed glyph image into SVG format.

    Steps:
        1. Save temporarily (required by VTracer)
        2. Convert to SVG
        3. Load SVG into memory
        4. Delete temporary files

    Args:
        input_image (np.ndarray): Preprocessed glyph image

    Returns:
        str: SVG content as string
    """

    if input_image is None:
        raise ValueError("input_image is None")

    img = input_image.copy()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
        temp_img_path = temp_img.name
        cv2.imwrite(temp_img_path, img)

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as temp_svg:
        temp_svg_path = temp_svg.name

    try:
        vectorize_image(temp_img_path, temp_svg_path)

        with open(temp_svg_path, "r") as f:
            svg_content = f.read()

    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        if os.path.exists(temp_svg_path):
            os.remove(temp_svg_path)

    return svg_content