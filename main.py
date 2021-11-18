from fastapi import FastAPI, HTTPException
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

model = PretrainedModel(config_app)

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
    age: int
    gender: str
    localization: str

@app.post("/predict/")
def predict(input: InputData):
    try:
        input.img = utils.base64ToPILImage(input.img)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid image base64 string value!")

    return model.predict(input)

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=config_app["server"]["ip_address"], 
        port=int(config_app["server"]["port"]), 
        reload=True,
        reload_includes=["app.yml"],
        reload_excludes=["test.py"]
    )