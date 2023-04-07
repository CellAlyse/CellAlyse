import pandas as pd
from skimage.transform import probabilistic_hough_line
from skimage.measure import label, regionprops
from skimage import measure
from scipy.ndimage import median_filter
import numpy as np
import streamlit as st
import cv2

def average_metrics(metrics):
    keys = metrics[0].keys()

    sum_vals = {key: 0 for key in keys}

    for m in metrics:
        for key in keys:
            if isinstance(m[key], (int, float)):
                sum_vals[key] += m[key]

    num_metrics = len(metrics)
    avg_metrics = {}
    for key in keys:
        if isinstance(metrics[0][key], (int, float)):
            avg_metrics[key] = sum_vals[key] / num_metrics

    return avg_metrics


def normalize_metrics(metrics):
    keys = metrics[0].keys()

    max_vals = {key: 0 for key in keys}

    for m in metrics:
        for key in keys:
            if isinstance(m[key], (int, float)):
                max_vals[key] = max(max_vals[key], m[key])

    for m in metrics:
        for key in keys:
            if isinstance(m[key], (int, float)):
                m[key] /= max_vals[key]

    return metrics

def remove_small_objects(mask, min_size):
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    max_area = 0
    for obj in props:
        max_area = max(max_area, obj.bbox_area)

    min_area = min_size * max_area
    for obj in props:
        if obj.bbox_area < min_area:
            mask[labeled_mask == obj.label] = 0

    return mask

def remove_overlapping_objects(seg_mask):
    labeled_mask = label(seg_mask)

    regions = regionprops(labeled_mask)

    non_overlap_mask = np.zeros_like(seg_mask, dtype=np.bool)

    for region in regions:
        minr, minc, maxr, maxc = region.bbox

        if np.any(non_overlap_mask[minr:maxr, minc:maxc]):
            continue

        non_overlap_mask[minr:maxr, minc:maxc] = (labeled_mask[minr:maxr, minc:maxc] == region.label)

    return non_overlap_mask
def compute_cell_metrics(mask, pixel_size, cell_type):
    if cell_type=='rbc':
        st.image(mask, use_column_width=True)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=4)
    # st.image(mask, caption="Dilated Mask", use_column_width=True)
    if cell_type == 'wbc':
        mask = remove_small_objects(mask, 0.2)
        st.image(mask, caption="Removed Small Objects", use_column_width=True)
        # remove overlapping objects
        #  st.image(mask, caption="Removed Overlapping Objects", use_column_width=True)
        # mask = remove_overlapping_objects(mask)


    pixel_size = float(pixel_size)
    conversion_factor = pixel_size ** 2

    labeled_mask = label(mask)

    props = regionprops(labeled_mask)


    metrics = []

    for obj in props:
        area = obj.bbox_area

        physical_area = area

        centroid = np.array(obj.centroid) * pixel_size

        obj_metrics = {}

        if st.session_state['Fläche']: # Fläche
            obj_metrics['Fläche'] = physical_area * conversion_factor

        if st.session_state['Umfang']: # Umfang
            obj_metrics['Umfang'] = obj.perimeter * pixel_size

        if st.session_state['Durchmesser']: # Durchmesser
            obj_metrics['Durchmesser'] = obj.equivalent_diameter * pixel_size

        if st.session_state['Orientierung']: # Orientierung
            obj_metrics['Orientierung'] = obj.orientation

        if st.session_state['Exzentrizität']:
            obj_metrics['Exzentrizität'] = obj.eccentricity

        metrics.append(obj_metrics)

    return metrics

