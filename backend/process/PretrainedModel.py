import tensorflow as tf
from backend.config.config import get_config
config_app = get_config()

class PretrainedModel:
    _instance = None
    def __new__(cls, cfg=None, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PretrainedModel, cls).__new__(cls, *args, **kwargs)

            #1. Load model image
            cls.resnet_FC = tf.keras.models.load_model(cfg['resnet_FC'])
        return cls._instance
