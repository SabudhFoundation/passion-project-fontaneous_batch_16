import sys
from pathlib import Path

from vectorization.preprocessing import process_glyph_image
from vectorization.vectorization import process_vectorization


def generate_svgs(data, font_name, svg_root):
    """
    Generate SVG files from glyph image data and save them in a structured folder.

    This function processes a dataset of glyph images, converts each image into a
    vectorized SVG using the glyph processing pipeline, and stores the results in a
    font-specific directory.

    Workflow:
        1. Extract valid items from input dataset (using "best" entries).
        2. Iterate through each glyph image.
        3. Convert image into processed glyph format using `process_glyph_image`.
        4. Vectorize glyph into SVG using `process_vectorization`.
        5. Save SVG file with corresponding label name.
        6. Track progress and count successfully saved SVGs.

    Args:
        data (dict): Nested dictionary containing glyph data. Expected format:
                     {
                         category: {
                             "best": [
                                 {"label": str, "img": np.ndarray},
                                 ...
                             ]
                         }
                     }
        font_name (str): Name of the font used to create output subfolder.
        svg_root (str or Path): Root directory where SVG folders will be created.

    Returns:
        Path: Path object pointing to the folder containing generated SVG files."""
    
    svg_folder = Path(svg_root) / font_name
    svg_folder.mkdir(parents=True, exist_ok=True)

    items = []
    for sub in data:
        if "best" in data[sub]:
            items.extend(data[sub]["best"])

    total = len(items)
    saved = 0

    for i, item in enumerate(items, 1):
        label = item["label"]
        img = item["img"]

        print(f"\r   Processing {i}/{total} → {label}", end="")
        sys.stdout.flush()

        glyph = process_glyph_image(img, f"{label}.png")
        if glyph is None:
            continue

        svg = process_vectorization(glyph)

        path = svg_folder / f"{label}.svg"
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)

        saved += 1

    print()
    print(f"[SVG] Generated {saved}/{total}")

    return svg_folder