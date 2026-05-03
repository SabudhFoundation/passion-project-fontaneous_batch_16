"""Streamlit app for the OCR to TTF workflow."""

import os
import sys
import shutil

import streamlit as st
import cv2
import numpy as np

import preprocessing_data.preprocessing as prep
from segmentation.Segmentation import process_segemntation
from char_ocr_labeling.main import run_ocr_pipeline


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(SRC_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SELECTED_DIR = os.path.join(PROCESSED_DIR, "selected_chars")
SVG_ROOT = os.path.join(PROCESSED_DIR, "svg")
TTF_DIR = os.path.join(PROCESSED_DIR, "ttf")

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

try:
    from vectorization.preprocessing import process_glyph_image
    from vectorization.vectorization import process_vectorization
    vectorization_error = None
except Exception as e:
    process_glyph_image = None
    process_vectorization = None
    vectorization_error = str(e)

try:
    from ttf_generation.ttf_final import build_font
    font_build_error = None
except Exception as e:
    build_font = None
    font_build_error = str(e)

st.set_page_config(page_title="OCR Pipeline", layout="wide")


def ensure_bgr(img):
    """Return a BGR image for segmentation."""
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def get_file_stem(filename):
    """Return the filename without extension."""
    return os.path.splitext(filename)[0]


def get_file_key(filename, file_bytes):
    """Build a unique key for the uploaded file."""
    return f"{filename}_{len(file_bytes)}_{file_bytes[:16].hex()}"


def get_safe_name(name):
    """Convert the file name into a safe font name."""
    safe_chars = []

    for char in name.lower():
        if char.isalnum():
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    safe_name = "".join(safe_chars).strip("_")

    if not safe_name:
        safe_name = "sample_font"

    if safe_name[0].isdigit():
        safe_name = f"font_{safe_name}"

    return safe_name


def ensure_data_folders():
    """Create the app data folders if needed."""
    folders = [
        DATA_DIR,
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        SELECTED_DIR,
        SVG_ROOT,
        TTF_DIR,
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def clear_folder(folder_path):
    """Delete all files and folders inside a folder."""
    if not os.path.exists(folder_path):
        return

    for name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, name)

        if os.path.isdir(item_path):
            shutil.rmtree(item_path, ignore_errors=True)
        else:
            try:
                os.remove(item_path)
            except OSError:
                pass


def clear_temp_folders():
    """Clear temporary workflow folders."""
    clear_folder(RAW_DIR)
    clear_folder(INTERIM_DIR)
    clear_folder(SELECTED_DIR)
    clear_folder(SVG_ROOT)


def delete_old_ttf():
    """Delete the previous TTF file."""
    clear_folder(TTF_DIR)


def reset_pipeline_state():
    """Reset the workflow session state."""
    st.session_state.selected = {}

    for key in [
        "file_key",
        "file_name",
        "input_image",
        "preprocessed_image",
        "char_crops",
        "ocr_result",
        "ttf_path",
    ]:
        st.session_state.pop(key, None)


def save_image(path, img):
    """Save an image to disk."""
    cv2.imwrite(path, img)


def is_supported_label(label):
    """Return True for labels supported by font generation."""
    return label.startswith("capital_") or label.startswith("small_") or label.isdigit()


def show_sample_crops(char_crops):
    """Show a small preview of detected crops."""
    cols = st.columns(6)

    for i, crop in enumerate(char_crops[:18]):
        with cols[i % 6]:
            st.image(crop, width=90)


def run_pipeline(uploaded):
    """Run preprocessing, segmentation and OCR."""
    file_bytes = uploaded.getvalue()
    file_name = get_file_stem(uploaded.name)

    img_array = np.frombuffer(file_bytes, np.uint8)
    input_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if input_image is None:
        raise ValueError("Could not read the uploaded image.")

    save_image(os.path.join(RAW_DIR, uploaded.name), input_image)

    preprocessed_image = prep.preprocess_single(input_image)
    save_image(
        os.path.join(INTERIM_DIR, f"{file_name}_preprocessed.png"),
        preprocessed_image,
    )

    seg_input = ensure_bgr(preprocessed_image)
    char_crops = process_segemntation(seg_input)

    if not char_crops:
        raise ValueError("No character crops detected.")

    ocr_result = run_ocr_pipeline(char_crops)

    if not ocr_result["groups"]:
        raise ValueError("No valid OCR results.")

    st.session_state["file_key"] = get_file_key(uploaded.name, file_bytes)
    st.session_state["file_name"] = file_name
    st.session_state["input_image"] = input_image
    st.session_state["preprocessed_image"] = preprocessed_image
    st.session_state["char_crops"] = char_crops
    st.session_state["ocr_result"] = ocr_result
    st.session_state["selected"] = {}
    st.session_state.pop("ttf_path", None)


def save_selected_svgs(selected, font_name):
    """Save selected glyphs as PNG and SVG files."""
    clear_folder(SELECTED_DIR)

    svg_dir = os.path.join(SVG_ROOT, font_name)
    os.makedirs(svg_dir, exist_ok=True)
    clear_folder(svg_dir)

    saved_count = 0
    skipped_count = 0

    for label in sorted(selected.keys()):
        if not is_supported_label(label):
            skipped_count += 1
            continue

        item = selected[label]
        img = item["img"]

        png_path = os.path.join(SELECTED_DIR, f"{label}.png")
        save_image(png_path, img)

        glyph_image = process_glyph_image(img, f"{label}.png")
        if glyph_image is None:
            skipped_count += 1
            continue

        svg_text = process_vectorization(glyph_image)
        svg_path = os.path.join(svg_dir, f"{label}.svg")

        with open(svg_path, "w", encoding="utf-8") as file:
            file.write(svg_text)

        saved_count += 1

    return saved_count, skipped_count


def generate_ttf_file(selected, file_name):
    """Generate the final TTF file."""
    if vectorization_error is not None:
        raise RuntimeError(f"Vectorization is unavailable: {vectorization_error}")

    if font_build_error is not None:
        raise RuntimeError(f"TTF generation is unavailable: {font_build_error}")

    font_name = get_safe_name(file_name)
    saved_count, skipped_count = save_selected_svgs(selected, font_name)

    if saved_count == 0:
        raise RuntimeError("No supported glyphs were selected for TTF generation.")

    ttf_path = os.path.join(TTF_DIR, f"{font_name}.ttf")

    if os.path.exists(ttf_path):
        os.remove(ttf_path)

    build_font(SVG_ROOT, [font_name], ttf_path, font_name)
    clear_temp_folders()

    return ttf_path, saved_count, skipped_count


def init_session_state():
    """Prepare folders and session state."""
    ensure_data_folders()

    if "selected" not in st.session_state:
        st.session_state.selected = {}


def show_upload_page():
    """Render the upload page."""
    st.title("Upload Image")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is None:
        st.info("Upload an image to start the workflow.")
        st.stop()

    file_bytes = uploaded.getvalue()
    file_key = get_file_key(uploaded.name, file_bytes)

    if st.session_state.get("file_key") != file_key:
        delete_old_ttf()
        clear_temp_folders()
        reset_pipeline_state()

        with st.spinner("Running preprocessing, segmentation and OCR labeling..."):
            try:
                run_pipeline(uploaded)
            except Exception as e:
                st.error(str(e))
                st.stop()

    st.image(st.session_state["input_image"], channels="BGR", width="stretch")

    st.subheader("Preprocessed Image")
    st.image(st.session_state["preprocessed_image"], clamp=True, width="stretch")

    st.success(f"{len(st.session_state['char_crops'])} character crops detected")

    st.subheader("Sample Crops")
    show_sample_crops(st.session_state["char_crops"])

    st.success("Processing complete. Go to Select OCR Images.")


def show_selection_page():
    """Render the OCR selection page."""
    st.title("Select OCR Images")

    if "ocr_result" not in st.session_state:
        st.warning("Upload an image first.")
        st.stop()

    groups = st.session_state["ocr_result"]["groups"]

    if not groups:
        st.error("No OCR labels found.")
        st.stop()

    st.info(f"Labels found: {len(groups)}")
    st.info(f"Selected labels: {len(st.session_state.selected)}")

    for label in sorted(groups.keys()):
        st.subheader(label)

        candidates = groups[label]
        cols = st.columns(6)

        for i, item in enumerate(candidates[:18]):
            with cols[i % 6]:
                st.image(item["img"], clamp=True, width=90)
                st.caption(f"score: {item['score']:.2f}")

                if st.button("Select", key=f"{label}_{i}"):
                    st.session_state.selected[label] = item

        if label in st.session_state.selected:
            st.success(f"Selected for {label}")

            if st.button("Clear selection", key=f"clear_{label}"):
                del st.session_state.selected[label]
                st.rerun()


def show_download_page():
    """Render the TTF download page."""
    st.title("Download TTF")

    if "ocr_result" not in st.session_state:
        st.warning("Upload an image first.")
        st.stop()

    selected = st.session_state.selected
    file_name = st.session_state.get("file_name", "sample_font")

    st.info(f"Selected glyphs: {len(selected)}")

    if not selected:
        st.warning("Select at least one glyph on the Select OCR Images page.")
        st.stop()

    if st.button("Generate TTF"):
        with st.spinner("Running vectorization and TTF generation..."):
            try:
                ttf_path, saved_count, skipped_count = generate_ttf_file(
                    selected,
                    file_name,
                )
            except Exception as e:
                st.error(str(e))
            else:
                st.session_state["ttf_path"] = ttf_path
                st.success(f"TTF ready with {saved_count} glyphs.")

                if skipped_count:
                    st.info(f"Skipped {skipped_count} unsupported glyphs.")

                st.info("Temporary files were cleaned. Only the TTF file remains.")

    ttf_path = st.session_state.get("ttf_path")

    if ttf_path and os.path.exists(ttf_path):
        with open(ttf_path, "rb") as file:
            st.download_button(
                "Download TTF",
                file.read(),
                file_name=os.path.basename(ttf_path),
                mime="font/ttf",
            )

        st.caption(ttf_path)


init_session_state()

page = st.sidebar.radio(
    "Go to",
    ["Upload Image", "Select OCR Images", "Download TTF"],
)

if page == "Upload Image":
    show_upload_page()
elif page == "Select OCR Images":
    show_selection_page()
elif page == "Download TTF":
    show_download_page()

