from streamlit_cropper import st_cropper
from apps.cbc import *


def cam():
    lottie_progress = load_lottifile("style/87081-blood-cell-morph.json")
    img_file_buffer = st.camera_input("Bild aufnehmen", key="cam")
    cell_type = st.selectbox(
        "Blutzellentyp", ("Rote Blutzellen", "Weiße Blutzellen", "Plättchen")
    )

    crop_me = st.sidebar.checkbox("Bild zuschneiden")
    if crop_me:
        realtime_update = True
        box_color = "#a9dc76"
        aspect_choice = st.sidebar.radio(
            label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"]
        )
        aspect_dict = {
            "1:1": (1, 1),
            "16:9": (16, 9),
            "16:9": (16, 9),
            "2:3": (2, 3),
            "Free": None,
        }
        aspect_ratio = aspect_dict[aspect_choice]

    if cell_type == "Rote Blutzellen":
        cell_type = "rbc"
    elif cell_type == "weiße Blutzelle":
        cell_type = "wbc"
    else:
        img = Image.open(f"{output_directory}/temp.jpg")
        cell_type = "plt"

    if img_file_buffer is not None:
        if crop_me:
            # convert img_file_buffer to PIL image
            img_file_buffer = Image.open(img_file_buffer)

            img = img_file_buffer
            cropped_img = st_cropper(
                img,
                realtime_update=realtime_update,
                box_color=box_color,
                aspect_ratio=aspect_ratio,
            )
            img_file_buffer = cropped_img
            _ = img_file_buffer.thumbnail((224, 224))
            st.image(img_file_buffer, use_column_width=True)

            # save img_file_buffer as a temporary file
            

        if st.button("Analyse starten"):
            prepare_upload(img_file_buffer, cam=crop_me)
            with st_lottie_spinner(lottie_progress, key="progress", loop=True):
                process(
                    img_file_buffer,
                    cell_type=cell_type,
                    cht=True,
                    ccl=True,
                    distancetransform=True,
                    upload=True,
                )

            st.experimental_memo.clear()
            os.remove(f"{output_directory}/temp.jpg")


def test():
    # create a unit test
    pass