from helper.functions import *
from apps.model import *

import streamlit as st
import requests
import json
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner


def load_lottifile(filepath: str):
    with open(filepath) as f:
        return json.load(f)


def cbc():
    lottie_progress = load_lottifile("style/87081-blood-cell-morph.json")

    st.sidebar.markdown("# Blutzellentyp")
    cell_type = st.sidebar.selectbox("", ("rote Blutzellen", "weiße Blutzelle", "Plättchen"))

    if cell_type == "rote Blutzellen":
        cell_type = "rbc"
    elif cell_type == "weiße Blutzelle":
        cell_type = "wbc"
    else:
        cell_type = "plt"

    st.sidebar.markdown("___")
    st.sidebar.markdown("# Methoden")
    cht = st.sidebar.checkbox("Circle Hough Transform")
    ccl = st.sidebar.checkbox("Connected Component Labeling")
    distancetransform = st.sidebar.checkbox("Distance Transform")
    st.sidebar.markdown("___")
    if cht and st.sidebar.checkbox("Parameter verändern"):
        minRadius = st.sidebar.slider("Minimaler Radius", 0, 100, value=28, key="minRadius")
        maxRadius = st.sidebar.slider("Maximaler Radius", 0, 100, value=55, key="maxRadius")
        minDist = st.sidebar.slider("Minimale Distanz", 0, 100, value=33, key="minDist")
        st.sidebar.markdown("___")
    else:
        minRadius = 28
        maxRadius = 55
        minDist = 33

    option = st.sidebar.selectbox("Upload oder Testbild?", ("Testbild", "Upload"))

    if option == "Testbild":
        st.markdown("# Bild auswählen")
        image = st.radio("", ("Testbild 1", "Testbild 2", "Testbild 3", "Testbild 4", "Testbild 5"))
        st.image(f"storage/images/bloodcount/{image}.jpg", use_column_width=True)
        upload = False

    else:
        image = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg", "tiff", "bmp"])

        if image is not None:
            prepare_upload(image)
            st.image(f'{output_directory}/temp.jpg', use_column_width=True)
            upload = True
            # clear cache
            st.experimental_memo.clear()

    if cell_type == 'rbc':
        out_img = 'edge_mask.png'
    else:
        out_img = 'mask.png'

    if st.button("Analyse starten") and image is not None:
        with st_lottie_spinner(lottie_progress, key="progress", loop=True):
            process(image, cell_type, cht, ccl, distancetransform, upload=upload, minRadius=minRadius, maxRadius=maxRadius,
                    minDist=minDist)

def process(image, cell_type, cht, ccl, distancetransform, upload=False, minRadius=28, maxRadius=55, minDist=33,
            maxDist=0):
    if upload:
        image = st_predict(f'{output_directory}/temp.jpg', cell_type)
    else:
        image = st_predict(f"storage/images/bloodcount/{image}.jpg", cell_type)


    # convert array to cv2 image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #st.image(image, use_column_width=True, clamp=True)


    if cell_type == 'rbc':
        out_img = 'edge_mask.png'
    else:
        out_img = 'mask.png'

    image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if cht:
        hough_transform(image, cell_type, minDist=minDist, maxRadius=maxRadius, minRadius=minRadius)
    if ccl:
        component_labeling(image)
    if distancetransform:
        threshold = stthreshold(image)
        # threshold = cv2.normalize(src=threshold, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        stcount( threshold, cell_type)

    if upload == False:
        st.write

