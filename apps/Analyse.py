from streamlit_lottie import st_lottie_spinner
import random

from helper.svm import *


def load_lottifile(filepath: str):
    with open(filepath) as f:
        return json.load(f)


def analyse():
    lottie_progress = load_lottifile("style/91877-data-analysis.json")

    st.sidebar.markdown("# Datensatz analyse")
    st.sidebar.markdown("___")

    model_name = st.sidebar.radio("", ("Raabin", "LISC", "BCCD"), index=0)

    st.markdown("# Bild aus dem BCCD Datensatz")
    st.markdown("___")
    st.image(f"storage/BCCD/{random.choice(os.listdir('storage/BCCD'))}", use_column_width=True)
    if st.button("Komplette Analyse starten"):
        with st_lottie_spinner(lottie_progress, key="progress", loop=True):
            dataset_prediction("BCCD", model_name)
