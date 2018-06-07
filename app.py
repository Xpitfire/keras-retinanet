import config_accessor as cfg

from flask import Flask, request, Response, jsonify

import traceback

import core
import misc
import logging

from model_encoder import ResponseEncoder
from models import Content
from services import classify_content, add_to_blacklist, remove_from_blacklist

logger = logging.getLogger('celum.app')

app = Flask(__name__)


def handle_request(content):
    try:
        classification_results = classify_content(content)
        encoder = ResponseEncoder(classification_results)
        return encoder.to_json()
    except:
        err = traceback.format_exc()
        json_response = jsonify({'exception': 'Server endpoint not responding! Please try again later.'})
        status = 500
        logger.error(err)
        return Response(json_response, status=status, mimetype='application/json')


@app.route('/classify/assets', methods=['POST'])
def classify_assets():
    json_content = request.get_json()
    content = Content(json_content)
    return handle_request(content)


@app.route('/blacklist/<string:asset_id>', methods=['DELETE'])
def blacklist_id(asset_id):
    ret = add_to_blacklist(asset_id)
    return Response(status=200 if ret == 0 else 404)


@app.route('/blacklist/undo/<string:asset_id>', methods=['GET'])
def undo_blacklist_id(asset_id):
    ret = remove_from_blacklist(asset_id)
    return Response(status=200 if ret == 0 else 404)


@app.route('/classify', methods=['GET'])
@misc.jsonp
def classify():
    id_ = request.args.get('id')
    url = request.args.get('url')
    if not id_:
        id_ = 'dummy'
    content = misc.classify_get_req_to_content(id_, url)
    return handle_request(content)


@app.before_first_request
def initialize():
    core.initialize_similarity_index()
    core.initialize_blacklist()
    core.initialize_elastic_search()
    core.initialize_retinanet()
    core.initialize_extraction_model()
    core.initialize_cron_job()


if __name__ == '__main__':
    core.initialize_logging()
    logger.info('Server app started!')
    app.run(host=cfg.resolve(cfg.RETINANET_SERVER, cfg.host),
            port=cfg.resolve_int(cfg.RETINANET_SERVER, cfg.port),
            debug=True)
