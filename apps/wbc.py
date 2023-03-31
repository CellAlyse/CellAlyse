import time

from helper.functions import *
from helper.svm import *

output_directory = "storage/tmp"


def wbc():
    global custom
    option_wbc = st.sidebar.radio("Optionen", ("Segmentieren", "Erkennen"))
    auto = False
    if option_wbc == "Erkennen":
        st.sidebar.markdown("___")
        model_name = st.sidebar.radio(
            "Modell auswählen", ("Raabin", "LISC", "BCCD"), index=0
        )
        auto = st.sidebar.checkbox("Automatische Bearbeitung", value=False)
        st.sidebar.markdown("___")
    else:
        st.sidebar.markdown("___")
        _bbox = st.sidebar.checkbox("Bounding Box", value=False)
        _further_analysis = st.sidebar.checkbox("Weitere Analyse", value=False)

    upload = st.sidebar.selectbox("Upload oder Testbild?", ("Testbild", "Upload"))

    if upload == "Testbild":
        st.markdown("# Bild auswählen")

        image = st.radio(
            "", ("Testbild 1", "Testbild 2", "Testbild 3", "Testbild 4", "Testbild 5")
        )
        st.image(f"storage/images/classification/{image}.jpg", use_column_width=True)
        custom = False
    elif upload == "Upload":
        st.markdown("# Bild hochladen")
        image = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg", "bmp"])
        if image is not None:
            prepare_upload(image)
            st.image(f"{output_directory}/temp.jpg", use_column_width=True)
            custom = True

    if st.button("Analyse starten") and image is not None:
        if option_wbc == "Segmentieren":
            segment_image(image, upload, _bbox, _further_analysis)
            


        else:
            predict_image(image, model_name, custom, auto)
            
            if upload == "Testbild":
                # get the image name. The name is the number at the end of the string
                img_name = image.split(" ")[1]
                
                st.markdown("___")
                st.markdown(f"Bei dem Testbild handelt es sich um ein: {show_true(int(img_name))}")

def show_true(img_id):
    if img_id == 1:
        return "Neutrophil"
    elif img_id == 2:
        return "Eosinophil"
    elif img_id == 3:
        return "Monozyt"
    elif img_id == 4:
        return "Lymphozyt"
    elif img_id == 5:
        return "Basophil"
    else:
        return "Keine Zelle"


def segment_image(img, upload, _bbox=False, _anal=False):
    if upload=="Upload":
        img = f"{output_directory}/temp.jpg"
    else:
        img = f"storage/images/classification/{img}.jpg"

    # load image, save relevant information, display image and relevant information
    image = cv2.imread(img)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if _bbox:
        bbox(image)
    elif _anal:
        st.write(f"Anzahl der Kerne (Nur bei Eosinophilen): {analyse_eosinophil(image)}")
        st.write(f"Anzahl der Körner (Nur bei Basophilen): {count_granules(image)}")
    else:
        nucleus, cnvx, roi = segmentation(image)
        st.image(nucleus, use_column_width=True)


def predict_image(img, model, upload, auto=False):
    if upload:
        img = f"{output_directory}/temp.jpg"
        if auto:
            large_image(cv2.imread(img), model_name=model)
            return
    else:
        img = f"storage/images/classification/{img}.jpg"

    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # st.image(img, use_column_width=True)
    start = time.time()
    prediction = predict_svm(img, model, x_train=model)
    end = time.time()

    run_time = round(end - start, 3)

    st.markdown(
        f"##### Blutzelle: <span style='color:#96b34a'>{read(prediction)}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"##### Laufzeit: <span style='color:#96b34a'>{run_time}</span> Sekunden",
        unsafe_allow_html=True,
    )


def read(prediction):
    switcher = {
        1: "Neutrophil",
        2: "Lymphozyt",
        3: "Monozyt",
        4: "Eosinophil",
        5: "Basophil",
    }
    return switcher.get(prediction, "Invalid")
