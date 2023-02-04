from streamlit_lottie import st_lottie_spinner
from PIL import Image

from helper.svm import *


icon = Image.open("style/favicon.ico")
st.set_page_config(
    page_title="CellAlyse",
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="expanded",
)
hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
               height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def load_lottifile(filepath: str):
    with open(filepath) as f:
        return json.load(f)



lottie_progress = load_lottifile("style/91877-data-analysis.json")

st.sidebar.markdown("# Datensatz analyse")
st.sidebar.markdown("___")

model_name = st.sidebar.radio("", ("Raabin", "LISC", "BCCD"), index=0)

st.markdown("# Bild aus dem BCCD Datensatz")
st.markdown("___")
st.image("storage/BCCD/mono_2_4392.jpeg", use_column_width=True)

if st.button("Komplette Analyse starten"):
    with st_lottie_spinner(lottie_progress, key="progress", loop=True):
        dataset_prediction("BCCD", model_name)
