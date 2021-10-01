# from flask import request
# from flask_restful import Resource
from backend.config.config import get_config
from backend.process.PretrainedModel import PretrainedModel
from backend.utils.preprocess import preprocess_image_input
import platform
import base64
import io
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2

config_app = get_config()
models = PretrainedModel()

# class SKIN_DETECTION(Resource):
#     def __init__(self):
#         Resource.__init__(self)
#     def post(self):

def skin_detection(data):
    ls_param = ['image']
    data = dict(data)
    # -------------------- Check missing params of request -------------------- #
    for ele in ls_param:
        if ele not in data or not data[ele]:
            # THROW ERROR because of not enough param
            return {'result': 'miss information'}

    if platform.python_version() > "3.9.0":   
        byteImgIO = io.BytesIO()
        byteImg = Image.open(data['image'])
        byteImg.save(byteImgIO, "PNG")
        byteImgIO.seek(0)
        byteImg = byteImgIO.read()

        # Non test code
        dataBytesIO = io.BytesIO(byteImg)
        image = Image.open(dataBytesIO)
    else:
        img_bin = base64.b64decode(data['image'])
        buf = io.BytesIO(img_bin)
        image = Image.open(buf)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(224,224), interpolation=cv2.INTER_AREA)
    # ----- predict
    class_names = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
    'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']
    image = preprocess_image_input(np.array(image))
    image = image.reshape(1,224,224,3)
    res = models.resnet_FC.predict(image)
    return {'result': class_names[np.argmax(res)]}