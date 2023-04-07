import base64
import json
import stack_data
from PIL import Image
import plotly.express as px
from scipy.stats import stats
from streamlit_lottie import st_lottie_spinner

from helper.svm import segmentation
from helper.model import *
from helper.functions import prepare_upload
from helper.advanced_metrics import *


def load_lottifile(filepath: str):
    with open(filepath) as f:
        return json.load(f)


def create_Layout():
    metrics = ["Fläche", "Umfang", "Durchmesser", "Orientierung", "Exzentrizität"]
    cell_type = st.sidebar.selectbox(
        "", ("rote Blutzellen", "weiße Blutzelle", "Plättchen"),
    )
    if cell_type == "rote Blutzellen":
        cell_type = "rbc"
    elif cell_type == "weiße Blutzelle":
        cell_type = "wbc"
    else:
        cell_type = "plt"
    if cell_type=="rbc":
        pixelProPicoMeter = st.sidebar.number_input("Pixelgröße in µm", value=0.092, key="pixelProPicoMeter", step=0.001, format="%.3f")
    elif cell_type=="wbc":
        pixelProPicoMeter = st.sidebar.number_input("Pixelgröße in µm", value=0.120, key="pixelProPicoMeter", step=0.001, format="%.3f")
    else:
        pixelProPicoMeter = st.sidebar.text_input("Pixelgröße in µm", value=0.007, key="pixelProPicoMeter")
    metrics = [st.sidebar.checkbox(metric, value=True, key=metric) for metric in metrics]
    st.warning(
        "Bitte beachten Sie, dass die Eingabe der Pixelgröße in µm nur eine grobe Schätzung ist. Die tatsächliche Pixelgröße kann sich von der Eingabe unterscheiden.")
    metrics.append(pixelProPicoMeter)
    return metrics, cell_type


def get_mask(image, cell_type):
    st.write(f"Cell type: {cell_type}")
    lottie_progress = load_lottifile("style/138949-analysis-bar-with-text.json")
    with st_lottie_spinner(lottie_progress, key="progress"):
        if cell_type == "rbc" or cell_type == "plt":
            mask = st_predict(image, cell_type)
        else:
            mask, _, _ = segmentation(cv2.imread(image))
    return mask


def get_image(cell_type):
    image = st.file_uploader(
        "Bild hochladen", type=["jpg", "png", "jpeg", "tiff", "bmp"]
    )
    if image is not None and st.button("Analyse starten"):
        # st.cache_data.clear()
        st.image(image, use_column_width=True, clamp=True)
        prepare_upload(image)
        mask = get_mask(f"{output_directory}/temp.jpg", cell_type)
        # st.image(mask, use_column_width=True, clamp=True)
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
    metrics, cell_type = create_Layout()
    mask = get_image(cell_type)
    if mask is not None:
        st.write(f"Mertiken {metrics}")
        metriken = compute_cell_metrics(mask, metrics[-1], cell_type)
        metriken_L = pd.DataFrame(metriken)
        metriken_L = metriken_L[(np.abs(stats.zscore(metriken_L)) < 3).all(axis=1)]

        df = pd.DataFrame(metriken_L)
        normalized = (df - df.mean()) / df.std()
        normalized = normalized.fillna(0)


        display_data(metriken_L, normalized)
        st.sidebar.markdown("---")
        if st.sidebar.button("Cache leeren"):
            st.cache_data.clear()
            st.cache_resource.clear()


def display_data(metriken, normalized):
    data = export_data(metriken, normalized)

    st.subheader("Durchschnittswerte")
    st.table(metriken.mean())

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
    st.dataframe(normalized, use_container_width=True)

    st.subheader("Alle Werte")
    st.dataframe(metriken, use_container_width=True)

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
