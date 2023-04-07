import streamlit

from helper.advanced_metrics import compute_cell_metrics
from apps.wbc import *
import plotly.express as px
import PIL
import matplotlib.pyplot as plt

@st.cache_data
def predict_batch(images, model):
    pred = []
    for i in images:
        pred.append(predict_svm(convert_Pil_to_cv2(Image.open(i)), model, model))
    return pred
def folder():
    images = st.file_uploader("Bilder hochladen", type=["jpg", "png", "jpeg", "bmp"], accept_multiple_files=True)
    model = st.sidebar.radio("Modell auswählen", ("Raabin", "LISC", "BCCD"), index=0)
    pixel_per_um = st.sidebar.number_input("Pixelgröße in µm", value=0.120, key="pixelProPicoMeter", step=0.001, format="%.3f")
    predicitons = []
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resources.clear()
    if images is not None and st.button("Analyse starten"):
        predicitons = predict_batch(images, model)
        for i in range(len(predicitons)):
            predicitons[i] = read(predicitons[i])

        metrics = []
        pred_metrics = []
        for i in range(len(predicitons)):
            mask, _, _ = segmentation(convert_Pil_to_cv2(Image.open(images[i])))
            metric = compute_wbc_metrics(mask, pixel_per_um)
            metrics.append(metric)
            pred_metrics.append((predicitons[i], metric))

        predicitons = dict((i, predicitons.count(i)) for i in predicitons)
        analyse_advanced_metrics(predicitons, metrics)
        analyse(predicitons)

def convert_Pil_to_cv2(image):
    image_jj = image.convert("RGB")
    opencvImage = cv2.cvtColor(np.array(image_jj), cv2.COLOR_RGB2BGR)
    return opencvImage

def analyse_advanced_metrics(predictions, metrics):
    cells = []
    for pred, count in predictions.items():
        for i in range(count):
            for j in metrics[i]:
                j_with_cell_type = {"Cell-type": pred}
                j_with_cell_type.update(j)
                cells.append(j_with_cell_type)

    df = pd.DataFrame(cells)
    st.dataframe(df, use_container_width=True)
    plot_advanced_metrics(df)

def plot_advanced_metrics(df):
    fig = px.scatter_3d(df, x="Cell-type", y="Fläche", z="Formfaktor", color="Cell-type", opacity=0.5)
    st.plotly_chart(fig)

    fig = px.box(df, x="Cell-type", y="Fläche", color="Cell-type")
    st.plotly_chart(fig)

def analyse(predictions: dict):
    df = pd.DataFrame.from_dict(predictions, orient="index", columns=["Anzahl"])

    # create a histogram using plotly
    fig = px.bar(df, x=df.index, y="Anzahl", color=df.index, color_discrete_sequence=px.colors.qualitative.Dark24)
    fig.update_layout(
        title="Histogramm",
        xaxis_title="Klasse",
        yaxis_title="Anzahl",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    st.plotly_chart(fig)

    st.dataframe(df, use_container_width=True)
def prepare_upload_folder(image_in, identifier):
    image_jj = image_in.convert("RGB")
    opencvImage = cv2.cvtColor(np.array(image_jj), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_directory}/{identifier}.jpg", opencvImage)