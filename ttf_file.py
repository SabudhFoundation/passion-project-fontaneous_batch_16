"""
====================================================================
MODULE: SVG to TTF Font Generation Pipeline (FontForge Engine)
====================================================================

Overview:
---------
This module implements a complete and reproducible pipeline for
generating a TrueType Font (TTF) from structured SVG glyph datasets.

The system performs:
- Dataset loading from multiple folders
- Unicode mapping from filenames
- Multi-candidate glyph scoring
- Best glyph selection per character
- Vector import into FontForge
- Geometric normalization and alignment
- Vector cleanup and contour correction
- Spacing normalization
- Font-wide optimization and validation
- Final TTF export

Dependencies:
------------
- fontforge
- psMat
- pathlib
"""

import fontforge
import psMat
from pathlib import Path


# ============================================================
# PROJECT PATH CONFIGURATION 
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BASE_FOLDER = BASE_DIR / "output_vectors_bestchars2"
OUTPUT_DIR = BASE_DIR / "LLASSTFONTS"
OUTPUT_TTF = OUTPUT_DIR / "Final_Fonts.ttf"

FOLDERS = ["kid1", "kid2", "kid3"]

FONT_NAME = "Final_Fonts"

UPM = 1000
ASCENDER = 800
DESCENDER = -200

X_HEIGHT = 500
CAP_HEIGHT = 700


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def filename_to_codepoint(filename: str):
    """
    Convert SVG filename into a Unicode codepoint.

    Expected filename format:
        'a_1.svg' → 'a' → U+0061
        '7_2.svg' → '7' → U+0037

    Parameters:
    ----------
    filename : str
        Name of SVG file.

    Returns:
    -------
    int or None
        Unicode codepoint if valid character, else None.
    """
    stem = Path(filename).stem
    base = stem.split("_")[0]
    return ord(base) if len(base) == 1 else None


def score_glyph(glyph):
    """
    Compute quality score for a glyph candidate.

    This scoring function evaluates multiple geometric properties
    to select the best SVG representation among duplicates.

    Evaluation Metrics:
    ------------------
    - Aspect ratio consistency
    - Stroke size balance
    - Shape complexity
    - Structural stability

    Parameters:
    ----------
    glyph : fontforge.glyph
        Temporary glyph used for scoring.

    Returns:
    -------
    float
        Higher value indicates better glyph quality.
    """
    bb = glyph.boundingBox()

    if bb == (0, 0, 0, 0):
        return -999

    xmin, ymin, xmax, ymax = bb
    w = xmax - xmin
    h = ymax - ymin

    if w == 0 or h == 0:
        return -999

    aspect_ratio = w / h

    return (
        -abs(aspect_ratio - 0.6)
        + min(w, h) / 100
        - abs(len(glyph.foreground) - 2)
        - abs(w - h) / 200
    )


def normalize_glyph(glyph, cp):
    """
    Normalize glyph geometry and align it to font baseline.

    This ensures consistent typographic scaling across all characters,
    including uppercase and lowercase glyphs.

    Parameters:
    ----------
    glyph : fontforge.glyph
        Glyph object to normalize.
    cp : int
        Unicode codepoint of the character.

    Returns:
    -------
    None
    """
    bb = glyph.boundingBox()

    w = bb[2] - bb[0]
    h = bb[3] - bb[1]

    if w == 0 or h == 0:
        return

    char = chr(cp)
    target_height = CAP_HEIGHT if char.isupper() else X_HEIGHT

    scale = target_height / h
    glyph.transform(psMat.scale(scale))

    bb2 = glyph.boundingBox()
    glyph.transform(psMat.translate(-bb2[0], DESCENDER - bb2[1]))


def apply_spacing(glyph):
    """
    Apply uniform horizontal spacing to a glyph.

    This function ensures consistent side bearings across all glyphs
    for visually balanced font rendering.

    Parameters:
    ----------
    glyph : fontforge.glyph
        Glyph object to modify.

    Returns:
    -------
    None
    """
    bb = glyph.boundingBox()

    w = bb[2] - bb[0]

    if w <= 0:
        glyph.width = 400
        return

    glyph.left_side_bearing = 50
    glyph.right_side_bearing = 50
    glyph.width = int(w + 100)


def clean_glyph(glyph):
    """
    Clean and optimize glyph vector data.

    This function performs geometric cleanup operations such as:
    - Removing overlapping paths
    - Simplifying curves
    - Adding extrema points
    - Normalizing contour direction
    - Filtering noisy contours
    - Rounding coordinates for stability

    Parameters:
    ----------
    glyph : fontforge.glyph
        Glyph object to clean.

    Returns:
    -------
    None
    """
    try:
        glyph.removeOverlap()
        glyph.simplify(2.0, ("ignoreextrema", "smoothcurves"))
        glyph.addExtrema()

        glyph.canonicalStart()
        glyph.canonicalContours()

        layer = glyph.foreground
        cleaned = fontforge.layer()

        for contour in layer:
            if len(contour) > 10:
                cleaned += contour

        glyph.foreground = cleaned

        glyph.round()
        glyph.removeOverlap()
        glyph.correctDirection()

    except Exception as e:
        print(f"[clean_glyph error] {e}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_font(base_folder, output_ttf, font_name):
    """
    Generate a TrueType Font (TTF) from SVG glyph dataset.

    This is the core pipeline that orchestrates:
    dataset loading → glyph scoring → best selection → normalization →
    cleaning → spacing → validation → final font export.

    Parameters:
    ----------
    base_folder : Path
        Root directory containing SVG glyph datasets.
    output_ttf : Path
        Output file path for generated TTF font.
    font_name : str
        Name of the font family.

    Returns:
    -------
    None
    """
    font = fontforge.font()

    font.fontname = font_name.replace(" ", "")
    font.familyname = font_name
    font.fullname = font_name

    font.encoding = "UnicodeFull"
    font.em = UPM
    font.ascent = ASCENDER
    font.descent = abs(DESCENDER)

    # .notdef glyph (fallback character)
    notdef = font.createChar(-1, ".notdef")
    pen = notdef.glyphPen()
    pen.moveTo((50, 50))
    pen.lineTo((550, 50))
    pen.lineTo((550, 750))
    pen.lineTo((50, 750))
    pen.closePath()
    notdef.width = 600

    char_map = {}

    # Load dataset (folder-based SVG collection)
    for folder_name in FOLDERS:
        folder_path = Path(base_folder) / folder_name

        if not folder_path.exists():
            continue

        for svg_file in folder_path.glob("*.svg"):
            cp = filename_to_codepoint(svg_file.name)
            if cp:
                char_map.setdefault(cp, []).append(svg_file)

    print(f"Collected {len(char_map)} characters")

    # Select best glyph per character
    for cp, svg_list in char_map.items():

        best_score = -999
        best_file = None

        for svg_file in svg_list:
            temp_font = fontforge.font()
            temp_glyph = temp_font.createChar(cp)

            try:
                temp_glyph.importOutlines(str(svg_file))
                temp_glyph.simplify(2.0, ("ignoreextrema", "smoothcurves"))
            except:
                continue

            score = score_glyph(temp_glyph)

            if score > best_score:
                best_score = score
                best_file = svg_file

        if not best_file:
            continue

        glyph = font.createChar(cp)

        try:
            glyph.importOutlines(str(best_file))
            glyph.simplify(2.0, ("ignoreextrema", "smoothcurves"))
            glyph.addExtrema()
            glyph.removeOverlap()
        except:
            continue

        normalize_glyph(glyph, cp)
        clean_glyph(glyph)
        apply_spacing(glyph)

        glyph.autoHint()

    # Fill missing ASCII characters
    for cp in range(32, 127):
        if cp not in char_map:
            font.createChar(cp).width = 400

    font.createChar(32).width = 300

    # Global font optimization
    font.selection.all()
    font.canonicalStart()
    font.canonicalContours()

    font.simplify(1.5)
    font.addExtrema()
    font.round()
    font.removeOverlap()
    font.correctDirection()

    print("Running validation...")
    print("Validation code:", font.validate())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    font.generate(str(output_ttf))

    print("Font generation completed successfully!")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    """
    Entry point for executing font generation pipeline.
    """
    build_font(BASE_FOLDER, OUTPUT_TTF, FONT_NAME)


if __name__ == "__main__":
    main()