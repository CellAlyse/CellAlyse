import base64

import stack_data
from PIL import Image
import plotly.express as px
from scipy.stats import stats

from helper.svm import segmentation
from helper.model import *
from helper.functions import prepare_upload
from helper.advanced_metrics import *

def create_Layout():
    metrics = ["Fläche","Umfang", "Durchmesser", "Orientierung", "Exzentrizität", "Abweichungen", "Varianzen"]
    pixelProPicoMeter = st.sidebar.text_input("Pixelgröße in µm", value=0.1)
    metrics = [st.sidebar.checkbox(metric, value=True) for metric in metrics]
    metrics.append(pixelProPicoMeter)
    return metrics


def get_mask(image, cell_type):
    if cell_type is "rbc" or "plt":
        mask = st_predict(image, cell_type)
    else:
        mask = segmentation(image)
    return mask

def get_image():
    image = st.file_uploader(
        "Bild hochladen", type=["jpg", "png", "jpeg", "tiff", "bmp"]
    )
    if image is not None and st.button("Analyse starten"):
        st.cache_data.clear()
        st.image(image, use_column_width=True, clamp=True)
        prepare_upload(image)
        mask = get_mask(f"{output_directory}/temp.jpg", "rbc")
        st.image(mask, use_column_width=True, clamp=True)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.normalize(
            src=mask,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        mask = stthreshold(mask)
        return mask

def export_data(metriken, normalized):
    metriken = pd.DataFrame(metriken)
    normalized = pd.DataFrame(normalized)

    metriken = metriken.to_csv(index=False)
    normalized = normalized.to_csv(index=False)
    return metriken, normalized
def metric():
    metrics = create_Layout()
    mask = get_image()
    if mask is not None:
        metriken = compute_cell_metrics(mask, metrics[-1], metrics[:-1])
        metriken_L = pd.DataFrame(metriken)
        metriken_L = metriken_L[(np.abs(stats.zscore(metriken_L)) < 3).all(axis=1)]

        normalized = normalize_metrics(metriken)
        averages = average_metrics(metriken)

        display_data(metriken_L, normalized, averages)

def display_data(metriken, normalized, averages):
    data = export_data(metriken, normalized)

    st.subheader("Durchschnittswerte")
    st.table(averages)

    st.subheader("Histogramm")
    st.bar_chart(metriken)

    st.subheader("Scatterplot")


    fig = px.scatter(
        metriken,
        x=metriken.index,
        y=metriken.columns,
        title="Scatterplot-Global",
    )
    st.plotly_chart(fig)

    fig = px.scatter_matrix(
        metriken,
        dimensions=metriken.columns,
        title="Scattermatrix-Global",
    )
    st.plotly_chart(fig)


    st.subheader("Normalisierte Werte")
    st.table(normalized)

    st.subheader("Alle Werte")
    st.table(metriken)

    st.sidebar.markdown("---")
    st.sidebar.download_button(
            label="Normalisierte Werte Herunterladen",
            data=data[1],
            file_name="normalized.csv",
            mime="text/csv",
        )
    st.sidebar.download_button(
            label="Alle Werte Herunterladen",
            data=data[0],
            file_name="metriken.csv",
            mime="text/csv",
        )




