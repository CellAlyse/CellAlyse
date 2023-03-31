from apps.wbc import *
import PIL
import matplotlib.pyplot as plt


def folder():
    images = st.file_uploader("Bilder hochladen", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    model = st.sidebar.radio("Modell auswählen", ("Raabin", "LISC", "BCCD"), index=0)
    predicitons = []
    if images is not None:
        i = 0
        for image in images:
            prepare_upload_folder(Image.open(image), i)
            i += 1
        
        for i in glob.glob("storage/tmp/*.jpg"):
            pred = folder_predict2(i, model=model, x_train_path=model)
            predicitons.append(pred)
        
        st.write([read(i) for i in predicitons])

        # create a dataframe, where the amount of each class is shown
        names = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
        counts = [predicitons.count(i) for i in range(4)]
        df = pd.DataFrame({"Class": names, "Count": counts})
        st.dataframe(df)

        # create a altair bar chart
        chart = alt.Chart(df).mark_bar().encode(
            x="Count:Q",
            y="Class:N",
            color="Class:N",
            tooltip=["Count", "Class"]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

        # delete all images in tmp folder
        for i in glob.glob("storage/tmp/*.jpg"):
             os.remove(i)


def prepare_upload_folder(image_in, identifier):
    image_jj = image_in.convert("RGB")
    opencvImage = cv2.cvtColor(np.array(image_jj), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_directory}/{identifier}.jpg", opencvImage)

def folder_predict2(
    img_path, model="data/Raabin.pkl", x_train_path="images/svm/x_train.npy"):
    model = joblib.load(f"storage/models/svm/{model}.pkl")
    x_train = np.load(f"storage/models/svm/{x_train_path}_train.npy")
    img = cv2.imread(img_path)
    #st.image(img_path)
    ncl_detect, error, ftrs = feature_extractor(img=img, min_area=100)
    if ncl_detect:
        ftrs = np.array(ftrs).reshape(1, -1)
        # normalize feature using max-min way
        mn, mx = x_train.min(axis=0), x_train.max(axis=0)
        ftrs = (ftrs - mn) / (mx - mn)
        pred = model.predict(ftrs)
        return pred[0]
    else:
        return error