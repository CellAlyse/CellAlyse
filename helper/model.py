import glob
import os
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
import tensorflow as tf
# import K
from tensorflow.keras import backend as K
from skimage.feature import peak_local_max
from scipy import ndimage
import streamlit as st
# Set the number of threads to use for parallelism
tf.config.threading.set_inter_op_parallelism_threads(6)
tf.config.threading.set_intra_op_parallelism_threads(6)

# optimise tensorflow
tf.config.optimizer.set_jit(True)

input_shape = (188, 188, 3)
output_shape = (100, 100, 1)
padding = [100, 100]

output_directory = "storage/tmp"


def conv_bn(
    filters,
    model,
    kernel=(3, 3),
    activation="relu",
    strides=(1, 1),
    padding="valid",
    type="normal",
):
    """
    :param filters --> Anzahl der Filter für jede Convolution
    :param kernel --> Kernel größe
    :param activation --> Aktivierungsfunktion
    :param strides --> Strides
    :param padding --> kann valid, same sein
    :param type --> zeigt op es sich um eine Transpose oder Normale Convolution handelt

    :return model --> Das Model
    """
    if type == "transpose":
        kernel = (2, 2)
        strides = 2
        conv = tf.keras.layers.Conv2DTranspose(filters, kernel, strides, padding)(model)
    else:
        conv = tf.keras.layers.Conv2D(filters, kernel, strides, padding)(model)

    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation(activation)(conv)

    return conv


def max_pool(input):
    """
    Max pooling layer mit eigenen Parametern
    """
    return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(input)


def concatenate(input1, input2, crop):
    """
    This is a general concatenation function with custom parameters.
    """
    return tf.keras.layers.concatenate(
        [tf.keras.layers.Cropping2D(crop)(input1), input2]
    )

@st.cache_resource
def do_unet(cell_type="rbc"):
    """
    Dual UNet model mit angepasster anzahl an Layern
    siehe -> model.summary()
    """
    inputs = tf.keras.layers.Input((188, 188, 3))

    # enc
    filters = 32
    encoder1 = conv_bn(3 * filters, inputs)
    encoder1 = conv_bn(filters, encoder1, kernel=(1, 1))
    encoder1 = conv_bn(filters, encoder1)
    pool1 = max_pool(encoder1)

    filters *= 2
    encoder2 = conv_bn(filters, pool1)
    encoder2 = conv_bn(filters, encoder2)
    pool2 = max_pool(encoder2)

    filters *= 2
    encoder3 = conv_bn(filters, pool2)
    encoder3 = conv_bn(filters, encoder3)
    pool3 = max_pool(encoder3)

    filters *= 2
    encoder4 = conv_bn(filters, pool3)
    encoder4 = conv_bn(filters, encoder4)

    # dec
    filters /= 2
    decoder1 = conv_bn(filters, encoder4, type="transpose")
    decoder1 = concatenate(encoder3, decoder1, 4)
    decoder1 = conv_bn(filters, decoder1)
    decoder1 = conv_bn(filters, decoder1)

    filters /= 2
    decoder2 = conv_bn(filters, decoder1, type="transpose")
    decoder2 = concatenate(encoder2, decoder2, 16)
    decoder2 = conv_bn(filters, decoder2)
    decoder2 = conv_bn(filters, decoder2)

    filters /= 2
    decoder3 = conv_bn(filters, decoder2, type="transpose")
    decoder3 = concatenate(encoder1, decoder3, 40)
    decoder3 = conv_bn(filters, decoder3)
    decoder3 = conv_bn(filters, decoder3)

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="mask")(
        decoder3
    )

    if cell_type == "rbc":
        out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="edge")(
            decoder3
        )
        model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge))
    elif cell_type == "wbc" or cell_type == "plt":
        model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask))

    opt = tf.optimizers.Adam(learning_rate=0.0001)

    if cell_type == "rbc":
        model.compile(
            loss="mse", loss_weights=[0.1, 0.9], optimizer=opt, run_eagerly=True
        )
    elif cell_type == "wbc" or cell_type == "plt":
        model.compile(loss="mse", optimizer=opt, run_eagerly=True)
    return model


def load_image_list(img_files):
    """
    :param img_files --> Liste mit Bildern (enumerate)

    :return imgs --> Liste mit Bildern
    """
    imgs = []
    for image_file in img_files:
        imgs += [cv2.imread(image_file)]
    return imgs


def clahe_images(img_list):
    """
    Clahe für jedes Bild in der Liste.
    Kann auch ohne Funktion in der Pipeline ausgeführt werden TODO: Optimierung <<.

    GENOMMEN VON https://www.youtube.com/watch?v=XfDkg3z3BCg

    :param img_list --> Liste mit Bildern
    :return img_list --> the output ima
    """
    for i, img in enumerate(img_list):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        img_list[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img_list


def preprocess_data(imgs, padding=padding[1]):
    """
    Padding wird zu den Bildern, Masken, Edges hinzugefügt.

    :param imgs --> Liste mit Bildern
    :param padding --> Input Padding welches zu den Bildern, Masken, Edges hinzugefügt wird

    :return imgs --> Liste mit Bildern mit padding
    """
    imgs = [
        np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
        for img in imgs
    ]
    return imgs

@st.cache_data
def load_data(img_list, padding=padding[1]):
    """
    Lädt die Bilder und ist gleichzeitig für das Preprocessing zuständig.

    :param img_list --> Liste mit Bildern
    :param padding --> Padding für das Preprocessing

    :return imgs --> Preprocessed Bilder
    """
    imgs = load_image_list(img_list)
    imgs = clahe_images(imgs)
    return preprocess_data(imgs, padding=padding)


def slice(
    imgs, padding=padding[1], input_size=input_shape[0], output_size=output_shape[0]
):
    """
    Die Bilder werden zu sogenannten chips zerteilt.
    Chips sollten 256 x 256 Pixel groß sein.
    :param imgs --> Input Bilder
    :param padding --> Padding für die Bilder
    :param input_size --> Input shape
    :param output_size --> output shape

    :return list tuple (list, list, list) --> Numpyarray vom Output(0 = Bild, 1 = Maske, 2 = Rand chips)
    """
    img_chips = []

    center_offset = padding + (output_size / 2)
    for i, _ in enumerate(imgs):
        for x in np.arange(
            center_offset, imgs[i].shape[0] - input_size / 2, output_size
        ):
            for y in np.arange(
                center_offset, imgs[i].shape[1] - input_size / 2, output_size
            ):
                chip_x_l = int(x - (input_size / 2))
                chip_x_r = int(x + (input_size / 2))
                chip_y_l = int(y - (input_size / 2))
                chip_y_r = int(y + (input_size / 2))

                temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]

                temp_chip = temp_chip.astype(np.float32) * 2
                temp_chip /= 255
                temp_chip -= 1

                img_chips += [temp_chip]
    return np.array(img_chips)


def normalize(img):
    """
    Normalisiert die Bilder.
    :param img --> Input Bild

    :return np.array --> Normalisiertes Bild in Form eines Numpyarray
    """
    return np.array((img - np.min(img)) / (np.max(img) - np.min(img)))


def get_sizes(img, padding=padding[1], input=input_shape[0], output=output_shape[0]):
    """
    Berechnet die Original Dimensionen des Bildes.

    :param img --> Bild von dem die Dimensionen berechnet werden sollen
    :param padding --> Default Padding für das Test Dataset
    :param input --> input shape
    :param output --> output shape

    :return couple --> a couple which contains the image dimensions as in (x, y)
    """
    offset = padding + (output / 2)
    return [
        (
            len(np.arange(offset, img[0].shape[0] - input / 2, output)),
            len(np.arange(offset, img[0].shape[1] - input / 2, output)),
        )
    ]


def reshape(img, size_x, size_y):
    """
    Mithile der Funktion get_sizes wird das Bild zum Original Umgewandelt
    :param img --> Das Bild, was zum Original umgewandelt werden soll
    :param size_x --> Länge des input Bildes
    :param size_y --> Höhe des input Bildes

    :return img (numpy array) --> Das input Bild wir in der gewünschten Form umgewandelt
    """
    return img.reshape(size_x, size_y, output_shape[0], output_shape[0], 1)


def concat(imgs):
    """
    Alle output image chips werden zu einem vollem Bild zusammengfügt (Concatination)
    :param imgs --> Liste mit output image chips

    :return full_image --> Volles zusammengefügtes Bild
    """
    return cv2.vconcat([cv2.hconcat(im_list) for im_list in imgs[:, :, :, :]])

@st.cache_data
def st_predict(img="images/Bild1.jpg", cell_type="rbc", old=True):
    """
    Segmentiere das Bild mit aktuell NUR U-NET.
    :param img --> Das Bild, das segmentiert werden soll
    :param plot_blocks --> Whether to plot the feature maps of each block in the U-Net

    :return --> Das bild wird im Output Ordner gespeichertÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜ
    """

    test_img = sorted(glob.glob(img))

    if not test_img:
        print("Bild nicht gefunden!")

    # init
    model = do_unet(cell_type=cell_type)
    if old:
        model.load_weights(
            f"storage/models/nn/{cell_type}.h5", skip_mismatch=True, by_name=True
        )
    else:
        model.load_weights(
            f"storage/models/nn/new/{cell_type}.h5", skip_mismatch=True, by_name=True
        )

    img = load_data(test_img, padding=padding[0])

    img_chips = slice(
        img,
        padding=padding[1],
        input_size=input_shape[0],
        output_size=output_shape[0],
    )

    # Segmentiere die chips
    output = model.predict(img_chips)

    if cell_type == "rbc":
        new_mask_chips = np.array(output[0])
        new_edge_chips = np.array(output[1])
    elif cell_type == "wbc" or cell_type == "plt":
        new_mask_chips = np.array(output)

    # Reshape für das Zusammensetzen der Chips
    dimensions = [get_sizes(img)[0][0], get_sizes(img)[0][1]]

    # Lösche unnötige Dimensionen/Informationen
    new_mask_chips = reshape(new_mask_chips, dimensions[0], dimensions[1])


    # Zusammensetzen der Chips
    new_mask_chips = np.squeeze(new_mask_chips)

    # Verbinde zu einem Vollbild
    new_mask = concat(new_mask_chips)
    if cell_type != "rbc":
        return new_mask
    else:
        new_edge_chips = reshape(new_edge_chips, dimensions[0], dimensions[1])
        new_edge_chips = np.squeeze(new_edge_chips)
        new_edge = concat(new_edge_chips)
        return new_mask - new_edge

def stthreshold(img="edge.png", cell_type="rbc"):
    image = img

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu_threshold, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)  #
    return image

def hough_transform(
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
    
    if cell_type == "rbc":
        circles = cv2.HoughCircles(
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
        circles = cv2.HoughCircles(
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
    

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255), 2)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)

        # output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
        return output, len(circles)
    else:
        # output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
        return output, 0


def component_labeling(img="edge.png"):
    image = img
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels = cv2.connectedComponents(img)

    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    output = cv2.merge([label_hue, blank_ch, blank_ch])

    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)

    output[label_hue == 0] = 0

    st.image(output, caption="Component Labeling", use_column_width=True)
    st.write(f"Connected component labeling: {num_labels}")
    
def stcount(img="threshold_mask.png", cell_type="rbc"):
    image = img
    img = np.asarray(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if cell_type == "rbc":
        min_distance = 40
        threshold_abs = 25
        exclude_border = True
    elif cell_type == "wbc":
        min_distance = 61
        threshold_abs = 20
        exclude_border = False
    elif cell_type == "plt":
        min_distance = 20
        img = ndimage.binary_dilation(img)
        threshold_abs = 4
        exclude_border = False

    edt = ndimage.distance_transform_edt(img)

    coords = peak_local_max(
        edt,
        indices=True,
        num_peaks=2000,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        exclude_border=exclude_border,
        labels=img,
    )

    # print(coords[:, 1])
    canvas = np.ones(img.shape + (3,), dtype=np.uint8) * 255
    i = 255
    for c in coords:
        o_c = (int(c[1]), int(c[0]))
        cv2.circle(canvas, o_c, 20, (i, 0, 0), -1)
        i = i - 1

    # saving image after counting
    st.image(canvas, caption="Euclidian Distance Transform", use_column_width=True)
    st.write(f"Count: {len(coords)}")
