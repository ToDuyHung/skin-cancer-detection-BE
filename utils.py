import base64
from io import BytesIO
from PIL import Image
import yaml
import imutils
import cv2

def base64ToPILImage(b64str: str):
    try:
        im_bytes = base64.b64decode(b64str)
        im_file = BytesIO(im_bytes)
        return Image.open(im_file)
    except:
        raise ValueError

def get_config():
    with open('app.yml', encoding='utf-8') as cfgFile:
        config_app = yaml.safe_load(cfgFile)

    return config_app

def aspect_aware_resize(image, width, height, inter=cv2.INTER_AREA):
    h, w = image.shape[:2]
    dW = 0
    dH = 0

    if w < h:
        image = imutils.resize(image, width, inter=inter)
        dH = int((image.shape[0] - height) / 2.0)
    else:
        image = imutils.resize(image, height=height, inter=inter)
        dW = int((image.shape[1] - width) / 2.0)

    h, w = image.shape[:2]
    image = image[dH:h-dH, dW:w-dW]
    return cv2.resize(image, (width, height), interpolation=inter)