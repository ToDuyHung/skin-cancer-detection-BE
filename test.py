import base64
import requests

with open("test.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

x = requests.post(url='http://localhost:6000/skin_detection', data = {'image': encoded_string})
# x = x.json()
print(x.text)