import json
from skimage import measure
from scipy.ndimage import median_filter
import streamlit as st
from streamlit_lottie import st_lottie_spinner
import pandas as pd
from skimage.transform import probabilistic_hough_line

from helper.functions import *
from helper.model import *


def load_lottifile(filepath: str):
    with open(filepath) as f:
        return json.load(f)


def cbc():
    lottie_progress = load_lottifile("style/87081-blood-cell-morph.json")
    area, pixel_size = False, 0.5
    st.sidebar.markdown("# Blutzellentyp")
    cell_type = st.sidebar.selectbox(
        "", ("rote Blutzellen", "weiße Blutzelle", "Plättchen")
    )

    model_old = "Alt"

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
        minRadius = st.sidebar.slider(
            "Minimaler Radius", 0, 100, value=28, key="minRadius"
        )
        maxRadius = st.sidebar.slider(
            "Maximaler Radius", 0, 100, value=55, key="maxRadius"
        )
        minDist = st.sidebar.slider("Minimale Distanz", 0, 100, value=33, key="minDist")
        st.sidebar.markdown("___")
    else:
        minRadius = 28
        maxRadius = 55
        minDist = 33


    option = st.sidebar.selectbox("Upload oder Testbild?", ("Testbild", "Upload"))
    if option == "Testbild":
        st.markdown("# Bild auswählen")
        image = st.radio(
            "", ("Testbild 1", "Testbild 2", "Testbild 3", "Testbild 4", "Testbild 5")
        )
        st.image(f"storage/images/bloodcount/{image}.jpg", use_column_width=True)
        upload = False

    else:
        image = st.file_uploader(
            "Bild hochladen", type=["jpg", "png", "jpeg", "tiff", "bmp"]
        )

        if image is not None:
            prepare_upload(image)
            st.image(f"{output_directory}/temp.jpg", use_column_width=True)
            upload = True
            st.cache_data.clear()

    if cell_type == "rbc":
        out_img = "edge_mask.png"
    else:
        out_img = "mask.png"
    

    if st.button("Analyse starten") and image is not None:
        with st_lottie_spinner(lottie_progress, key="progress", loop=True):
            edge_mask= process(
                image,
                cell_type,
                cht,
                ccl,
                distancetransform,
                upload=upload,
                minRadius=minRadius,
                maxRadius=maxRadius,
                minDist=minDist,
                old=model_old
            )
def process(
    image,
    cell_type,
    cht,
    ccl,
    distancetransform,
    upload=False,
    minRadius=28,
    maxRadius=55,
    minDist=33,
    maxDist=0,
    old="Alt"
):
    if old == "Alt":
        model_old = True
    else:
        model_old = False
    if upload:
        image = st_predict(f"{output_directory}/temp.jpg", cell_type, old=model_old)
        st.image(image, use_column_width=True, clamp=True)
    else:
        image = st_predict(f"storage/images/bloodcount/{image}.jpg", cell_type,old=model_old)
        st.image(image, use_column_width=True, clamp=True)
    #cht_opt = None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # st.image(image, use_column_width=True, clamp=True)

    image = cv2.normalize(
        src=image,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    if cht:
        hough_transform(
            image, cell_type, minDist=minDist, maxRadius=maxRadius, minRadius=minRadius
        )
    if ccl:
        component_labeling(image)
    if distancetransform:
        threshold = stthreshold(image)
        stcount(threshold, cell_type)   
    return image

def compute_cell_metrics(mask, pixel_size):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    avg_sizes = []
    cell_sizes = []
    avg_perimeters = []
    for label in range(1, num_labels):
        label_mask = (labels == label).astype(np.uint8)

        area = stats[label, cv2.CC_STAT_AREA]
        label_dist_transform = label_mask * dist_transform

        avg_size = np.sum(label_dist_transform) * float(pixel_size) / area
        avg_sizes.append(avg_size)
        cell_sizes.append(int(area * pixel_size ** 2))

        perimeter = cv2.arcLength(np.array(np.where(label_mask == 1)).T, closed=True)
        avg_perimeter = perimeter * pixel_size / area
        avg_perimeters.append(avg_perimeter)

    print('label_dist_transform:', label_dist_transform.shape, label_dist_transform.dtype)
    print('pixel_size:', pixel_size, type(pixel_size))
    print('area:', area, type(area))

    avg_size = np.sum(label_dist_transform) * float(pixel_size) / area
    avg_perimeter = np.mean(avg_perimeters)
    std_size = np.std(cell_sizes)
    std_perimeter = np.std(avg_perimeters)

    return avg_size, cell_sizes, avg_perimeter, std_size, std_perimeter
