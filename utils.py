import base64
from io import BytesIO
from PIL import Image

def base64ToPILImage(b64str: str):
    im_bytes = base64.b64decode(b64str)
    im_file = BytesIO(im_bytes)
    return Image.open(im_file)