from PIL import Image


def prepare_upload(img):
    # save the image as a temporary file.
    image = Image.open(img)
    image.save("storage/tmp/temp.jpg")
    return img

