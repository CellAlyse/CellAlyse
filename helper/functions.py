from PIL import Image


def prepare_upload(img, cam=False):
    # save the image as a temporary file.
    if not cam:
        image = Image.open(img)
        image = image.convert("RGB")
        image.save("storage/tmp/temp.jpg")
    else:
        img.save("storage/tmp/temp.jpg")
        return None
    return img
