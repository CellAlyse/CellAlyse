
import cv2
import numpy as np


import streamlit as st
import pyhdust.images as phim
from scipy.spatial import ConvexHull
import joblib
from skimage import filters as fl


@st.experimental_singleton
def segmentation(img):
    """
    :param img: input rgb Bild
    :param min_area: Min Area so werden plätchen umgangen
    :return: binary vom Nukleus, binary of Konvexe Hülle, binary von REst vom Zytoplasma
    """
    org_img = img.copy()

    # Color balancing
    Gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    mean_gray = np.mean(Gray)
    mean_R = np.mean(R)
    mean_G = np.mean(G)
    mean_B = np.mean(B)

    R_ = R * (mean_gray / mean_R)
    G_ = G * (mean_gray / mean_G)
    B_ = B * (mean_gray / mean_B)

    R_[R_ > 255] = 255
    G_[G_ > 255] = 255
    B_[B_ > 255] = 255

    balance_img = np.zeros_like(org_img)
    balance_img[:, :, 0] = R_.copy()
    balance_img[:, :, 1] = G_.copy()
    balance_img[:, :, 2] = B_.copy()

    # balance_img = org_img.copy()
    cmyk = phim.rgb2cmyk(balance_img)
    _M = cmyk[:, :, 1]
    _K = cmyk[:, :, 3]

    _S = cv2.cvtColor(balance_img, cv2.COLOR_RGB2HLS_FULL)[:, :, 2]

    min_MS = np.minimum(_M, _S)
    a_temp = np.where(_K < _M, _K, _M)
    KM = _K - a_temp

    b_temp = np.where(min_MS < KM, min_MS, KM)
    min_MS_KM = min_MS - b_temp

    #    cv2.imshow('Step 1' , cv2.resize(Nucleus_img , (256 ,256)))

    # Step 2 :
    min_MS_KM = cv2.GaussianBlur(min_MS_KM, ksize=(5, 5), sigmaX=0)
    try:
        thresh2 = fl.threshold_multiotsu(min_MS_KM, 2)
        Nucleus_img = np.zeros_like(min_MS_KM)
        Nucleus_img[min_MS_KM >= thresh2] = 255
    except:
        print('try-Except')
        _M = cv2.GaussianBlur(_M, ksize=(5, 5), sigmaX=0)
        thresh2 = fl.threshold_multiotsu(_M, 2)
        Nucleus_img = np.zeros_like(_M)
        Nucleus_img[_M >= thresh2] = 255

    contours, _ = cv2.findContours(Nucleus_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pad_del = np.zeros_like(Nucleus_img)

    max_area = max(cv2.contourArea(contours[idx]) for idx in np.arange(len(contours)))
    for j in range(len(contours)):
        if cv2.contourArea(contours[j]) < (max_area / 10):
            cv2.drawContours(pad_del, contours, j, color=255, thickness=-1)
    Nucleus_img[pad_del > 0] = 0

    contours, _ = cv2.findContours(Nucleus_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _perimeter = 0
    for cnt in contours:
        _perimeter += cv2.arcLength(cnt, True)

    temp_points = np.argwhere(Nucleus_img == 255)
    Ncl_points = np.zeros_like(temp_points)
    Ncl_points[:, 0] = temp_points[:, 1]
    Ncl_points[:, 1] = temp_points[:, 0]
    _area = np.sum(Nucleus_img)

    cvx_hull = ConvexHull(Ncl_points)
    Cvx_area = cvx_hull.volume
    Cvx_prm = cvx_hull.area
    Verc = cvx_hull.vertices
    Corners = []
    for idx in range(len(Verc)):
        tempcol = Ncl_points[Verc[idx], 0]
        temprow = Ncl_points[Verc[idx], 1]
        Corners.append([tempcol, temprow])
    Corners = np.array(Corners)
    Corners = np.reshape(Corners, newshape=(Corners.shape[0], 1, 2))

    img_convex = np.zeros_like(Nucleus_img)
    cv2.drawContours(img_convex, [Corners], 0, color=255, thickness=-1)
    CVX_points = np.argwhere(img_convex == 255)

    img_ROC = img_convex - Nucleus_img

    return Nucleus_img, img_convex, img_ROC


def feature_extractor(img, min_area=100):
    """
    :param img: input rgb bild
    :param min_area: minimale Fläche der Weißen Blutzelle. Wird benutzt um Plätchen Segmentierung zu umgehen
    :return: binäres Bild vom Nukleus, binäres Bild der Konvexen Hülle, binäres Bild vom Rest vom Zytoplasma(ROC)
    """
    Ftr_List = []

    org_img = img.copy()
    img[:, :, 0] = org_img[:, :, 0].copy()
    img[:, :, 1] = org_img[:, :, 1].copy()
    img[:, :, 2] = org_img[:, :, 2].copy()

    Gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    mean_gray = np.mean(Gray)
    mean_R = np.mean(R)
    mean_G = np.mean(G)
    mean_B = np.mean(B)

    R_ = R * (mean_gray / mean_R)
    G_ = G * (mean_gray / mean_G)
    B_ = B * (mean_gray / mean_B)

    R_[R_ > 255] = 255
    G_[G_ > 255] = 255
    B_[B_ > 255] = 255

    balance_img = np.zeros_like(org_img)
    balance_img[:, :, 0] = R_.copy()
    balance_img[:, :, 1] = G_.copy()
    balance_img[:, :, 2] = B_.copy()

    # balance_img = org_img.copy()
    cmyk = phim.rgb2cmyk(balance_img)
    _M = cmyk[:, :, 1]
    _K = cmyk[:, :, 3]

    _S = cv2.cvtColor(balance_img, cv2.COLOR_RGB2HLS_FULL)[:, :, 2]

    min_MS = np.minimum(_M, _S)
    a_temp = np.where(_K < _M, _K, _M)
    KM = _K - a_temp

    b_temp = np.where(min_MS < KM, min_MS, KM)
    min_MS_KM = min_MS - b_temp
    min_MS_KM = cv2.GaussianBlur(min_MS_KM, ksize=(5, 5), sigmaX=0)

    try:
        thresh2 = fl.threshold_multiotsu(min_MS_KM, 2)
        Nucleus_img = np.zeros_like(min_MS_KM)
        Nucleus_img[min_MS_KM >= thresh2] = 255
    except:
        print('try-Except')
        _M = cv2.GaussianBlur(_M, ksize=(5, 5), sigmaX=0)
        thresh2 = fl.threshold_multiotsu(_M, 2)
        Nucleus_img = np.zeros_like(_M)
        Nucleus_img[_M >= thresh2] = 255

    contours, _ = cv2.findContours(Nucleus_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pad_del = np.zeros_like(Nucleus_img)

    max_area = max(cv2.contourArea(contours[idx]) for idx in np.arange(len(contours)))
    for j in range(len(contours)):
        if cv2.contourArea(contours[j]) < (max_area / 10):
            cv2.drawContours(pad_del, contours, j, color=255, thickness=-1)
    Nucleus_img[pad_del > 0] = 0

    contours, _ = cv2.findContours(Nucleus_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _perimeter = 0
    for cnt in contours:
        _perimeter += cv2.arcLength(cnt, True)

    temp_points = np.argwhere(Nucleus_img == 255)
    Ncl_points = np.zeros_like(temp_points)
    Ncl_points[:, 0] = temp_points[:, 1]
    Ncl_points[:, 1] = temp_points[:, 0]
    _area = np.sum(Nucleus_img)

    cvx_hull = ConvexHull(Ncl_points)
    Cvx_area = cvx_hull.volume
    Cvx_prm = cvx_hull.area
    Verc = cvx_hull.vertices
    Corners = []
    for idx in range(len(Verc)):
        tempcol = Ncl_points[Verc[idx], 0]
        temprow = Ncl_points[Verc[idx], 1]
        Corners.append([tempcol, temprow])
    Corners = np.array(Corners)
    Corners = np.reshape(Corners, newshape=(Corners.shape[0], 1, 2))

    img_convex = np.zeros_like(Nucleus_img)
    cv2.drawContours(img_convex, [Corners], 0, color=255, thickness=-1)
    CVX_points = np.argwhere(img_convex == 255)

    img_ROC = img_convex - Nucleus_img
    ROC_points = np.argwhere(img_ROC == 255)

    flag_empty = len(contours) > 0
    if not flag_empty:
        Error = '[Error 1]: Kein Umriss gefunden'
        print(Error)
        return False, Error, None

    if max_area <= min_area:
        Error = f'[ERROR 2]:Max area kleiner als min area {min_area}'
        print(Error)
        return False, Error, None

    Circularity = (_perimeter) ** 2 / (4 * 3.14 * _area)
    Convexity = (Cvx_prm / _perimeter)
    Solidity = (_area / Cvx_area)
    Shape_Features = np.array([Circularity, Convexity, Solidity])
    Ftr_List.extend([Circularity, Convexity, Solidity])
    if np.sum(img_convex - Nucleus_img) == 0:
        print('Convex Bild == Nukeus Bild')
        temp = [1] * 72
        temp.extend(Ftr_List)
        return True, None, np.array(temp)

    ALL_Channels = []
    ALL_Channels.append(balance_img[:, :, 0])  # channel R : index 0
    ALL_Channels.append(balance_img[:, :, 1])  # channel G : index 1
    ALL_Channels.append(balance_img[:, :, 2])  # channel B : index 2

    HSV = cv2.cvtColor(balance_img, cv2.COLOR_RGB2HSV)
    ALL_Channels.append(HSV[:, :, 0])  # channel H : index 3
    ALL_Channels.append(HSV[:, :, 1])  # channel S : index 4
    ALL_Channels.append(HSV[:, :, 2])  # channel V : index 5

    LAB = cv2.cvtColor(balance_img, cv2.COLOR_RGB2LAB)
    ALL_Channels.append(LAB[:, :, 0])  # channel L : index 6
    ALL_Channels.append(LAB[:, :, 1])  # channel A : index 7
    ALL_Channels.append(LAB[:, :, 2])  # channel BB : index 8

    YCrCb = cv2.cvtColor(balance_img, cv2.COLOR_RGB2YCrCb)
    ALL_Channels.append(YCrCb[:, :, 0])  # channel Y : index 9
    ALL_Channels.append(YCrCb[:, :, 1])  # channel Cr : index 10
    ALL_Channels.append(YCrCb[:, :, 2])  # channel Cb : index 11

    NCL_pxls_value = np.zeros(shape=(len(ALL_Channels), Ncl_points.shape[0]),
                              dtype=np.uint8)  # intensitität vom  Nukleus
    CVX_pxls_Value = np.zeros(shape=(len(ALL_Channels), CVX_points.shape[0]),
                              dtype=np.uint8)  # intensitität vom  Konvex
    ROC_pxls_Value = np.zeros(shape=(len(ALL_Channels), ROC_points.shape[0]), dtype=np.uint8)  # intensitität vom  ROC

    for ch in range(len(ALL_Channels)):
        p_roc, p_ncl = 0, 0
        for p in range(CVX_points.shape[0]):
            row, col = CVX_points[p, 0], CVX_points[p, 1]
            CVX_pxls_Value[ch, p] = ALL_Channels[ch][row, col]

            if Nucleus_img[row, col] == 255:
                NCL_pxls_value[ch, p_ncl] = ALL_Channels[ch][row, col]
                p_ncl += 1
            else:
                ROC_pxls_Value[ch, p_roc] = ALL_Channels[ch][row, col]
                p_roc += 1

    # durch und std für alle channels
    Ncl_mean_std = np.zeros(shape=(len(ALL_Channels), 2), dtype=np.float)
    Ncl_mean_std[:, 0] = np.mean(NCL_pxls_value, axis=1)
    Ncl_mean_std[:, 1] = np.std(NCL_pxls_value, axis=1)

    # durch und std für Konvex
    Cvx_mean_std = np.zeros(shape=(len(ALL_Channels), 2), dtype=np.float)
    Cvx_mean_std[:, 0] = np.mean(CVX_pxls_Value, axis=1)
    Cvx_mean_std[:, 1] = np.std(CVX_pxls_Value, axis=1)

    # durch und std für ROC
    Roc_mean_std = np.zeros(shape=(len(ALL_Channels), 2), dtype=np.float)
    Roc_mean_std[:, 0] = np.mean(ROC_pxls_Value, axis=1)
    Roc_mean_std[:, 1] = np.std(ROC_pxls_Value, axis=1)

    Ratio_Ncl2Cvx = np.reshape(np.divide(Ncl_mean_std, Cvx_mean_std), newshape=(len(ALL_Channels) * 2,))
    Ratio_Roc2Cvx = np.reshape(np.divide(Roc_mean_std, Cvx_mean_std), newshape=(len(ALL_Channels) * 2,))
    Ratio_Roc2Ncl = np.reshape(np.divide(Roc_mean_std, Ncl_mean_std), newshape=(len(ALL_Channels) * 2,))
    Color_Features = np.concatenate((Ratio_Ncl2Cvx, Ratio_Roc2Cvx))
    Color_Features = np.nan_to_num(Color_Features, nan=0, posinf=1)
    ALL_Features = np.concatenate((Color_Features, Shape_Features))

    return True, None, ALL_Features


def predict_svm(image, model='Raabin', x_train='Raabin'):
    model = joblib.load(f'storage/models/svm/{model}.pkl')
    x_train = np.load(f'storage/models/svm/{x_train}_train.npy')

    ncl, error, features = feature_extractor(image)

    if ncl:
        features = np.array(features).reshape(1, -1)
        # normalize feature using max-min way
        mn, mx = x_train.min(axis=0), x_train.max(axis=0)
        features = (features - mn) / (mx - mn)
        pred = model.predict(features)
        return pred[0]
    else:
        return error


@st.experimental_memo
def load_model(model_path, x_train):
    model = joblib.load(model_path)
    x_train = np.load(x_train)

    return model, x_train

def large_image(image, model_name):
    nuclei, _, _ = segmentation(image)

    # get bounding boxes of every nucleus
    contours, _ = cv2.findContours(nuclei, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]

    # display image with bounding boxes
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # st.image(image, caption=f'{model_name} - {len(boxes)} nuclei detected', use_column_width=True)
    st.write(model_name)
    if model_name == 'LISC':
        vertical = 87
        horizontal = 91
    elif model_name == 'Raabin':
        vertical = 184 // 2
        horizontal = 192 // 2
    else:
        vertical = 240 // 2
        horizontal = 320 // 2

    for box in boxes:
        x, y, w, h = box

        center_x = x + w // 2
        center_y = y + h // 2

        window_x = center_x - horizontal
        window_y = center_y - vertical

        if window_x < 0:
            window_x = 0
        if window_y < 0:
            window_y = 0
        if window_x + horizontal * 2 > image.shape[1]:
            window_x = image.shape[1] - horizontal * 2
        if window_y + vertical * 2 > image.shape[0]:
            window_y = image.shape[0] - vertical * 2

        window = image[window_y:window_y + vertical * 2, window_x:window_x + horizontal * 2]

        col1, col2 = st.columns(2)
        col1.image(window, use_column_width=True)
        col2.write(f'Blutzelle: {read(predict_svm(window, model_name))}')



def read(prediction):
    switcher = {
        1: "Neutrophil",
        2: "Lymphozyt",
        3: "Monozyt",
        4: "Eosinophil",
        5: "Basophil"
    }
    return switcher.get(prediction, "Invalid")