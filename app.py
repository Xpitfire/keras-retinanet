from flask import Flask, request, Response, jsonify

import traceback

import settings
from misc import jsonp
import logging

from services import classify_content

logger = logging.getLogger('celum.app')

app = Flask(__name__)


def handle_request(content):
    try:
        classification_results = classify_content(content)
        return jsonify(classification_results)
    except:
        err = traceback.format_exc()
        json_response = jsonify({'exception': 'Server endpoint not responding! Please try again later.'})
        status = 500
        logger.error(err)
        return Response(json_response, status=status, mimetype='application/json')


@app.route('/classify/assets', methods=['POST'])
def classify_assets():
    content = request.get_json()
    return handle_request(content)


@app.route('/classify', methods=['GET'])
@jsonp
def classify():
    url = request.args.get('url')
    content = {
        "assets": [
            {
                "asset-id": "<no-process-demo>",
                "url": url
            }
        ]
    }
    return handle_request(content)


@app.before_first_request
def initialize():
    settings.initialize_similarity_index()
    settings.initialize_elastic_search()
    settings.initialize_retinanet()
    settings.initialize_extraction_model()


if __name__ == '__main__':
    settings.initialize_settings()
    settings.initialize_logging()
    logger.info('Server app started!')
    app.run(host=settings.config['RETINANET_SERVER']['host'],
            port=int(settings.config['RETINANET_SERVER']['port']),
            debug=True)
