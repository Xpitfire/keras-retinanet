from flask import Flask, request, Response, jsonify

import traceback

from misc import jsonp
from classify import classify_urls

import logging
import settings

app = Flask(__name__)


def resolve_special_url_cmd(image_path):
    result_list = []
    for path in image_path:
        if 'file://' in path and path.endswith('.list'):
            path = path.replace('file://', '')
            content = [line.strip() for line in open(path, 'r').readlines()]
            result_list += content
        else:
            result_list.append(path)
    return result_list


@app.route('/classify', methods=['GET'])
@jsonp
def classify():
    try:
        image_path = request.args.get('url').split(';')
        logging.info(image_path)
        image_path = resolve_special_url_cmd(image_path)
        classification_results = classify_urls(image_path)
        return jsonify(classification_results)
    except:
        err = traceback.format_exc()
        json_response = jsonify({'exception': 'Server endpoint not responding! Please try again later.'})
        status = 500
        logging.error(err)
        return Response(json_response, status=status, mimetype='application/json')


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
