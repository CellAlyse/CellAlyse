import time
from helper.svm import *
from streamlit_lottie import st_lottie_spinner


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
    st.image("storage/BCCD/mono_2_4392.jpeg", use_column_width=True)

    if st.button("Komplette Analyse starten"):
        with st_lottie_spinner(lottie_progress, key="progress", loop=True):
            dataset_prediction("BCCD" ,model_name)