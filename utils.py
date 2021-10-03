import base64
from io import BytesIO
from PIL import Image
import yaml

def base64ToPILImage(b64str: str):
    im_bytes = base64.b64decode(b64str)
    im_file = BytesIO(im_bytes)
    return Image.open(im_file)

def get_config():
    with open('app.yml', encoding='utf-8') as cfgFile:
        config_app = yaml.safe_load(cfgFile)

    return config_app