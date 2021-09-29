from flask_restful import Api
from flask import Flask, request, g, jsonify
from backend.config.config import get_config
import logging
from logging.handlers import RotatingFileHandler
from time import strftime
import traceback
import time

app = Flask(__name__)
api = Api(app)

config_app = get_config()

logging.basicConfig(filename=config_app['log']['app'],
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

from backend.api.api_skin_detection import SKIN_DETECTION

api.add_resource(SKIN_DETECTION, '/skin_detection')

@app.before_request
def before_request():
    g.start = time.time()

@app.after_request
def after_request(response):
    end = time.time()
    diff = end - g.start
    timestamp = strftime('[%Y-%b-%d %H:%M]')
    if response.status_code == 200:
        logger.info('%s %s %s %s %s %s %s', timestamp, str(diff), request.remote_addr, request.method, request.scheme, response.status, response.get_json())
    else:
        logger.error('%s %s %s %s %s %s %s %s ', timestamp, str(diff), request.remote_addr, request.method, request.scheme,
                     request.full_path, response.status, response.get_json())
    return response


@app.errorhandler(Exception)
def exceptions(err):
    tb = traceback.format_exc()
    timestamp = strftime('[%Y-%b-%d %H:%M]')
    logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s', timestamp, request.remote_addr, request.method,
                 request.scheme, request.full_path, tb)
    return jsonify({"message": "Hệ thống xảy ra lỗi", "traceback": tb}), 500

handler = RotatingFileHandler(config_app['log']['request'], maxBytes=1000000000, backupCount=3)
logger = logging.getLogger('request')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

app.run(host=config_app['server']['ip_address'], port=config_app['server']['port'], debug=True, threaded=True)