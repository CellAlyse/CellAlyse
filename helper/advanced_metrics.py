import pandas as pd
from skimage.transform import probabilistic_hough_line
from skimage.measure import label, regionprops
from skimage import measure
from scipy.ndimage import median_filter
import numpy as np
import streamlit as st

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


def compute_cell_metrics(mask, pixel_size, options):
    pixel_size = float(pixel_size)

    labeled_mask = label(mask)

    props = regionprops(labeled_mask)

    metrics = []

    for obj in props:
        area = obj.bbox_area

        physical_area = area * pixel_size**2

        centroid = np.array(obj.centroid) * pixel_size

        obj_metrics = {}

        if options[0]: # Fläche
            obj_metrics['Fläche'] = physical_area

        if options[1]: # Umfang
            obj_metrics['Umfang'] = obj.perimeter * pixel_size

        if options[2]: # Durchmesser
            obj_metrics['Durchmesser'] = obj.equivalent_diameter * pixel_size

        if options[3]: # Orientierung
            obj_metrics['Orientierung'] = obj.orientation

        if options[4]: # Exzentrizität
            obj_metrics['Exzentrizität'] = obj.eccentricity

        metrics.append(obj_metrics)

    return metrics

