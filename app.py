from flask import Flask, request, Response, jsonify

import traceback

from classify import classify_content

import logging
import settings
from misc import jsonp

app = Flask(__name__)


def handle_request(content):
    try:
        classification_results = classify_content(content)
        return jsonify(classification_results)
    except:
        err = traceback.format_exc()
        json_response = jsonify({'exception': 'Server endpoint not responding! Please try again later.'})
        status = 500
        logging.error(err)
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
    settings.init_retinanet()


if __name__ == '__main__':
    settings.initialize_settings()
    settings.initialize_logging()
    logging.info('Server app started!')
    app.run(host=settings.config['RETINANET_SERVER']['host'],
            port=int(settings.config['RETINANET_SERVER']['port']),
            debug=True)
