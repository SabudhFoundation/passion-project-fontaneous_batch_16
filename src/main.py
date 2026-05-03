import streamlit as st
import cv2
import numpy as np
import os

import preprocessing_data.preprocessing as prep
from segmentation.Segmentation import process_segemntation
from char_ocr_labeling.main import run_ocr_pipeline
from char_ocr_labeling.saver import save_final_outputs

st.set_page_config(page_title="OCR Pipeline", layout="wide")


# =========================================
# UTILS
# =========================================
def ensure_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def get_file_stem(filename):
    return os.path.splitext(filename)[0]


# =========================================
# SESSION INIT
# =========================================
if "selected" not in st.session_state:
    st.session_state.selected = {}

# =========================================
# SIDEBAR
# =========================================
page = st.sidebar.radio("Go to", ["Input", "Selection", "Save"])


# =========================================
# INPUT PAGE
# =========================================
if page == "Input":

    st.title("Upload & Process")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded:
        file_bytes = uploaded.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        st.image(img, channels="BGR", width="stretch")

        # =========================
        # PREPROCESS
        # =========================
        with st.spinner("Running preprocessing..."):
            preprocessed = prep.preprocess_single(img)

        st.subheader("Preprocessed Image")
        st.image(preprocessed, channels="GRAY", width="stretch")

        # convert to 3-channel for segmentation
        seg_input = ensure_bgr(preprocessed)

        # =========================
        # SEGMENTATION
        # =========================
        with st.spinner("Running segmentation..."):
            char_crops = process_segemntation(seg_input)

        if not char_crops:
            st.error("No character crops detected.")
            st.stop()

        st.success(f"{len(char_crops)} character crops detected")

        # Preview crops
        st.subheader("Sample Crops")
        cols = st.columns(8)
        for i, crop in enumerate(char_crops[:24]):
            with cols[i % 8]:
                st.image(crop, width=80)

        # =========================
        # OCR PIPELINE
        # =========================
        with st.spinner("Running OCR labeling..."):
            ocr_result = run_ocr_pipeline(char_crops)

        st.session_state["ocr_result"] = ocr_result
        st.session_state["char_crops"] = char_crops
        st.session_state["file_name"] = get_file_stem(uploaded.name)

        # reset selection for new upload
        st.session_state.selected = {}

        st.success("Processing complete → Go to Selection")


# =========================================
# SELECTION PAGE
# =========================================
elif page == "Selection":

    st.title("Select Best Characters")

    if "ocr_result" not in st.session_state:
        st.warning("Run pipeline first")
        st.stop()

    groups = st.session_state["ocr_result"]["groups"]

    if not groups:
        st.error("No valid OCR results")
        st.stop()

    st.success(f"{len(groups)} labels detected")

    for label in sorted(groups.keys()):

        st.subheader(label)

        candidates = groups[label]

        cols = st.columns(8)

        for i, item in enumerate(candidates[:24]):

            with cols[i % 8]:
                st.image(item["img"], width=80)
                st.caption(f"{item['score']:.2f}")

                key = f"{label}_{i}"

                if st.button("Select", key=key):
                    st.session_state.selected[label] = item

        if label in st.session_state.selected:
            st.success(f"Selected for {label}")


# =========================================
# SAVE PAGE
# =========================================
elif page == "Save":

    st.title("Save Dataset")

    if "ocr_result" not in st.session_state:
        st.warning("Run pipeline first")
        st.stop()

    labeled = st.session_state["ocr_result"]["labeled"]
    selected = st.session_state.selected
    file_name = st.session_state.get("file_name", "sample")

    if not labeled:
        st.error("No labeled data")
        st.stop()

    st.info(f"Labeled samples: {len(labeled)}")
    st.info(f"Selected best: {len(selected)}")

    if st.button("Save Dataset"):

        data = {
            file_name: {
                "labeled": labeled,
                "best": list(selected.values())
            }
        }

        save_final_outputs(data)

        st.success("Dataset saved")

        st.markdown("""
        **Output folders:**
        - `final_dataset/`
        - `best_chars/`
        - `best_chars_inverted/`
        """)