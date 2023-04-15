import json
from skimage import measure
from scipy.ndimage import median_filter
import streamlit as st
from streamlit_lottie import st_lottie_spinner
import pandas as pd
from skimage.transform import probabilistic_hough_line
import random

from helper.functions import *
from helper.model import *


def load_lottifile(filepath: str):
    with open(filepath) as f:
        return json.load(f)


def cbc():
    # Load lottie progress bar
    lottie_progress = load_lottifile("style/87081-blood-cell-morph.json")

    area, pixel_size = False, 0.5
    cell_types = {
        "rote Blutzellen": "rbc",
        "weiße Blutzelle": "wbc",
        "Plättchen": "plt",
    }
    methods = {
        "Circle Hough Transform": "cht",
        "Connected Component Labeling": "ccl",
        "Distance Transform": "distancetransform",
    }
    min_radius = 28
    max_radius = 55
    min_dist = 33
    cht_opt = False
    option = st.sidebar.selectbox("Upload oder Testbild?", ("Testbild", "Upload"))

    # Sidebar options
    st.sidebar.markdown("# Blutzellentyp")
    cell_type = st.sidebar.selectbox("", list(cell_types.keys()))
    cell_type = cell_types[cell_type]

    #model_type = st.sidebar.selectbox("Modell", ("Alt", "Neu"))
    model_type = "Alt"
    st.sidebar.markdown("___")
    st.sidebar.markdown("# Methoden")
    method_checkboxes = {}
    for method in methods:
        method_checkboxes[method] = st.sidebar.checkbox(method)
    st.sidebar.markdown("___")

    if method_checkboxes["Circle Hough Transform"] and st.sidebar.checkbox("Parameter verändern"):
        if st.sidebar.checkbox("Optimieren"):
            cht_opt = True
            min_radius = 28
            max_radius = 55
            min_dist = 33
        else:
            min_radius = st.sidebar.slider(
                "Minimaler Radius", 0, 100, value=28, key="minRadius"
            )
            max_radius = st.sidebar.slider(
                "Maximaler Radius", 0, 100, value=55, key="maxRadius"
            )
            min_dist = st.sidebar.slider("Minimale Distanz", 0, 100, value=33, key="minDist")

    # Image options
    if option == "Testbild":
        st.markdown("# Bild auswählen")
        test_images = {
            f"Testbild {i}": f"storage/images/bloodcount/Testbild {i}.jpg"
            for i in range(1, 6)
        }
        image = st.radio("", list(test_images.keys()))
        st.image(test_images[image], use_column_width=True)
        upload = False

    else:
        st.markdown("# Bild hochladen")
        image = st.file_uploader(
            "Bild hochladen", type=["jpg", "png", "jpeg", "tiff", "bmp"]
        )

        if image is not None:
            prepare_upload(image)
            st.image(f"{output_directory}/temp.jpg", use_column_width=True)
            upload = True
            st.cache_data.clear()

    if st.button("Analyse starten") and image is not None:
        with st_lottie_spinner(lottie_progress, key="progress", loop=True):
            edge_mask = process(
                image,
                cell_type,
                method_checkboxes["Circle Hough Transform"],
                method_checkboxes["Connected Component Labeling"],
                method_checkboxes["Distance Transform"],
                upload=upload,
                minRadius=min_radius,
                maxRadius=max_radius,
                minDist=min_dist,
                old=model_type,
                cht_opt=cht_opt
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
        old="Alt",
        cht_opt=False
):
    if old == "Alt":
        model_old = True
    else:
        model_old = False
    if upload:
        image = st_predict(f"{output_directory}/temp.jpg", cell_type, old=model_old)
        st.image(image, use_column_width=True, clamp=True)
    else:
        image = st_predict(f"storage/images/bloodcount/{image}.jpg", cell_type, old=model_old)
        st.image(image, use_column_width=True, clamp=True)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.normalize(
        src=image,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    if cht:
        if cht_opt:
            circles, length = hough_transform_auto(img=image, cell_type=cell_type, minDist=minDist, maxRadius=maxRadius,
                                                   minRadius=minRadius)
            circles = np.stack((circles, circles, circles), axis=-1)
            print(f"Circles_Shape: {circles.shape}")
            st.image(circles, use_column_width=True, clamp=False)
            minDist, maxRadius, minRadius, param1, param2 = optimize_hough_transform(circles, image, cell_type)
            circles, length = hough_transform_auto(img=image, cell_type=cell_type, minDist=minDist, maxRadius=maxRadius,
                                                   minRadius=minRadius, param1=param1, param2=param2)
            st.image(circles, use_column_width=True, clamp=False)
            st.markdown(f"**Anzahl der Zellen: {length}**")
            st.markdown(
                f"**minDist: {minDist}, maxRadius: {maxRadius}, minRadius: {minRadius}, param1: {param1}, param2: {param2}**")
        else:
            circles, length = hough_transform_auto(img=image, cell_type=cell_type, minDist=minDist, maxRadius=maxRadius,
                                                   minRadius=minRadius)
            st.image(circles, use_column_width=True, clamp=False)
            st.markdown(f"**Anzahl der Zellen: {length}**")
    if ccl:
        component_labeling(image)
    if distancetransform:
        threshold = stthreshold(image)
        stcount(threshold, cell_type)
    return image


def hough_transform_score(params, img, mask):
    """
    This function computes the score of Hough Transform parameters for counting cells in a segmented image.
    """
    minDist, maxRadius, minRadius, param1, param2 = map(int, params)
    print(f"minDist: {minDist}, maxRadius: {maxRadius}, minRadius: {minRadius}, param1: {param1}, param2: {param2}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        1,
        minDist=int(minDist),
        maxRadius=int(maxRadius),
        minRadius=int(minRadius),
        param1=int(param1),
        param2=int(param2),
    )
    if circles is not None:
        error = np.sum((img_gray - mask) ** 2)
        return -1 / (error + 1)
    else:
        return -np.inf


def optimize_hough_transform(img, mask, cell_type):
    # Define the parameter bounds
    bounds = {
        "minDist": (33, 100),
        "maxRadius": (55, 80),
        "minRadius": (28, 40),
        "param1": (30, 50),
        "param2": (20, 50)
    }

    step_sizes = {
        "minDist": 1,
        "maxRadius": 2,
        "minRadius": 2,
        "param1": 5,
        "param2": 5
    }

    params = {}
    for param in bounds:
        params[param] = int(random.uniform(*bounds[param]))

    def score(params, img, mask):
        minDist = int(params['minDist'])
        maxRadius = int(params['maxRadius'])
        minRadius = int(params['minRadius'])
        param1 = int(params['param1'])
        param2 = int(params['param2'])

        # ray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        circles_score, _ = hough_transform_auto(mask, cv2.HOUGH_GRADIENT, param1=param1, minDist=minDist, param2=param2,
                                                minRadius=minRadius, maxRadius=maxRadius)
        if circles_score is not None:
            circles = np.round(circles_score[0, :]).astype("int")
            if len(circles) == 4:
                return 1.0 - (np.std([circles[0][0], circles[1][0], circles[2][0], circles[3][0]]) + np.std(
                    [circles[0][1], circles[1][1], circles[2][1], circles[3][1]])) / 2.0
        return 0.0

    current_score = score(params, img, mask)
    while True:
        new_params = dict(params)

        param_to_perturb = random.choice(list(bounds.keys()))

        new_params[param_to_perturb] += random.uniform(-step_sizes[param_to_perturb], step_sizes[param_to_perturb])

        new_params[param_to_perturb] = np.clip(new_params[param_to_perturb], *bounds[param_to_perturb])

        new_score = score(new_params, img, mask)

        if new_score > current_score:
            params = new_params
            current_score = new_score
        else:
            break

    return params["minDist"], params["maxRadius"], params["minRadius"], params["param1"], params["param2"]


def hough_transform_auto(
        img="edge.png",
        cell_type="rbc",
        minDist=33,
        maxRadius=55,
        minRadius=28,
        param1=30,
        param2=20,
):
    # Load the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles_len = None
    if cell_type == "rbc":
        circles_len = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            1,
            minDist=minDist,
            maxRadius=int(maxRadius),
            minRadius=int(minRadius),
            param1=param1,
            param2=param2,
        )
    elif cell_type == "wbc" or cell_type == "plt":
        circles_len = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            1,
            minDist=minDist,
            maxRadius=int(maxRadius),
            minRadius=int(minRadius),
            param1=param1,
            param2=param2,
        )
    output = img.copy()

    if circles_len is not None:
        circles = np.round(circles_len[0, :]).astype("int")
        for (x, y, r) in circles:
            # cv2.circle(output, (x, y), r, (0, 255), 2)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)

        return output, len(circles)
    else:
        return output, 0
