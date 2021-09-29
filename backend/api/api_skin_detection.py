from flask import request
from flask_restful import Resource
from backend.config.config import get_config
import platform
import base64
import io
from PIL import Image
config_app = get_config()

class SKIN_DETECTION(Resource):
    def __init__(self):
        Resource.__init__(self)
    def post(self):
        ls_param = ['image']
        # -------------------- Check missing params of request -------------------- #
        for ele in ls_param:
            if ele not in request.form or not request.form[ele]:
                # THROW ERROR because of not enough param
                return {'result': 'ERROR NOT ENOUGH PARAM'}

        # Add values to each of params - Dictionary
        input_data = {
            ele: request.form[ele]
            for ele in ls_param
        }
        if platform.python_version() > "3.9.0":   
            byteImgIO = io.BytesIO()
            byteImg = Image.open(input_data['image'])
            byteImg.save(byteImgIO, "PNG")
            byteImgIO.seek(0)
            byteImg = byteImgIO.read()

            # Non test code
            dataBytesIO = io.BytesIO(byteImg)
            image = Image.open(dataBytesIO).convert('RGB')
        else:
            img_bin = base64.b64decode(input_data['image'])
            buf = io.BytesIO(img_bin)
            image = Image.open(buf).convert('RGB')
        image.show()
        return input_data