from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

from MLModel import *
import utils

app = FastAPI()

config_app = utils.get_config()

logging.basicConfig(filename=config_app['log']['app'],
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

model = SimpleANN()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="http://localhost:.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

class InputData(BaseModel):
    img: str

@app.post("/predict")
def predict(input: InputData):
    img = utils.base64ToPILImage(input.img)
    return model.predict(img)

if __name__ == "__main__":
    uvicorn.run("main:app", host=config_app["server"]["ip_address"], port=int(config_app["server"]["port"]), reload=True)