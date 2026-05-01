import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import segmentation.Segmentation as seg
import preprocessing_data.preprocess as prep
st.set_page_config(page_title="Image Processor")

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Input", "Output"])

#Input Page 
if page == "Input":
    st.title("Input")


    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded:
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)

        file_bytes = uploaded.getvalue()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = prep.preprocess(img)
        segmentation_img = seg.process_segemntation(img)
    
    
# Output Page
elif page == "Output":
    st.title("Output")

    if "file_bytes" not in st.session_state:
        st.warning("No file processed yet. Go to the Input page first.")
    else:
        st.image(st.session_state["file_bytes"], caption="Processed Image", use_container_width=True)

        st.download_button(
            label="Download File",
            data=st.session_state["file_bytes"],
            file_name="output_" + st.session_state["file_name"],
            mime="image/png",
        )