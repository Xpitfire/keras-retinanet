import config_accessor as cfg

from flask import Flask, request, Response, jsonify

import traceback

import sys
import core
import misc
import logging

from model_encoder import ResponseEncoder
from models import Content
from services import classify_content, add_to_blacklist, remove_from_blacklist, reset_blacklist

logger = logging.getLogger('celum.app')

app = Flask(__name__)


default_error_message = 'Server endpoint not responding! Please check your request or try again later.'


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['exception'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def handle_request(content):
    try:
        classification_results = classify_content(content)
        encoder = ResponseEncoder(classification_results)
        return encoder.to_json()
    except:
        err = traceback.format_exc()
        logger.error(err)
        raise InvalidUsage('Error while processing request! Please try again later...')


def parse_post_req_content():
    try:
        json_content = request.get_json()
        content = Content(json_content)
        return content
    except:
        err = traceback.format_exc()
        logger.error(err)
        raise InvalidUsage('Could not parse request! Please check ids and request format.')


@app.route('/services/v1/classify', methods=['POST'])
def classify_assets():
    content = parse_post_req_content()
    max_requests = cfg.resolve_int(cfg.CLASSIFICATION, cfg.max_assets_per_request)
    if len(content.assets) > max_requests:
        raise InvalidUsage('Exceeded maximum number of assets ({}) per request!'.format(max_requests))
    return handle_request(content)


@app.route('/services/v1/classify', methods=['GET'])
@misc.jsonp
def classify():
    id_ = request.args.get('id')
    url = request.args.get('url')
    if not id_ or not url:
        raise InvalidUsage('Missing id or url parameter!')
    content = misc.classify_get_req_to_content(id_, url)
    return handle_request(content)


@app.route('/services/v1/blacklist/<string:asset_id>', methods=['DELETE'])
def blacklist_id(asset_id):
    ret = add_to_blacklist(asset_id)
    return Response(status=200 if ret == 0 else 404)


@app.route('/services/v1/blacklist/undo/<string:asset_id>', methods=['GET'])
def undo_blacklist_id(asset_id):
    ret = remove_from_blacklist(asset_id)
    return Response(status=200 if ret == 0 else 404)


@app.route('/services/v1/blacklist/init', methods=['GET'])
def init_blacklist():
    core.initialize_blacklist()
    return Response(status=200)


@app.route('/services/v1/blacklist/reset', methods=['DELETE'])
def blacklist_reset():
    ret = reset_blacklist()
    return Response(status=200 if ret == 0 else 404)


@app.route('/services/v1/shutdown', methods=['GET'])
def shutdown_hook():
    core.trigger_backup()
    sys.exit()


@app.route('/services/v1/index/init', methods=['GET'])
def init_similarity_index():
    core.initialize_elastic_search()
    return Response(status=200)


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
            debug=cfg.resolve_bool(cfg.RETINANET_SERVER, cfg.debug),
            threaded=cfg.resolve_bool(cfg.RETINANET_SERVER, cfg.threaded))
