import streamlit as st
import os
import os.path

import matplotlib.pyplot as plt
import numpy as np

import skimage.io
import skimage.morphology

import tensorflow as tf
import keras

import utils.metrics
import utils.cell_u_net
from PIL import Image
from config import config_vars

def setup(config_vars=None):
    partition = "validation"
    experiment_name = 'cell_u_net'
    config_vars = utils.dirtools.setup_experiment(config_vars, experiment_name)
    data_partitions = utils.dirtools.read_data_partitions(config_vars)

def predict(image):
        global probmap, pred, label

        imagebuffer = skimage.io.imread_collection(image)
        images = imagebuffer.concatenate()

        dim1 = images.shape[1]
        dim2 = images.shape[2]

        images = images.reshape((-1, dim1, dim2, 1))

        images = images / 255

        model = utils.cell_u_net.get_model_3_class(dim1, dim2)

        model.load_weights("models/model.hdf5")
        model.summary()
        predictions = model.predict(images, batch_size=1)
        print(f"Länge: {len(images)}")
        for i in range(len(images)):
            print(i)
            print(imagebuffer.files[i])
            filename = imagebuffer.files[i]
            filename = os.path.basename(filename)
            print(filename)

            probmap = predictions[i].squeeze()
            pred = utils.metrics.probmap_to_pred(probmap, config_vars["boundary_boost_factor"])
            skimage.io.imsave("storage/tmp/pred/" + filename, pred)
            label = utils.metrics.pred_to_label(pred, config_vars["cell_min_size"])
        return probmap, pred, label



def main():
    setup(config_vars)
    file_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if file_upload is not None:
        pil_img = Image.open(file_upload)
        pil_img.save("storage/tmp/tmp.png")
        image_path = "storage/tmp/tmp.png"
        st.image(pil_img, caption='Uploaded Image.', use_column_width=True)

        if st.button("Analyse"):
            probmap, pred, label = predict(image_path)
            st.image(Image.open("storage/tmp/pred/tmp.png"), clamp=True)
            st.image(probmap)
            st.image(label)

            for file in os.listdir("storage/tmp"):
                if os.path.isfile(os.path.join("tmp", file)):
                    os.remove(os.path.join("tmp", file))
