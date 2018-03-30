import redis
from flask import Flask
from flask import request, redirect
from flask import Response

import traceback

from misc import jsonp, jsonify
from models import Result
from classify import classify_urls

import logging

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='retinanet.log',
                    filemode='a')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

logging.info('Started redis host')


def resolve_special_url_cmd(image_path):
    result_list = []
    for path in image_path:
        if 'file://' in path and path.endswith('.list'):
            path = path.replace('file://','')
            content = [line.strip() for line in open(path, 'r').readlines()]
            result_list += content
        else:
            result_list.append(path)
    return result_list

@app.route("/classify", methods=['GET'])
@jsonp
def classify():
    try:
        image_path = request.args.get('url').split(';')
        logging.info(image_path)
        image_path = resolve_special_url_cmd(image_path)
        classification_results = classify_urls(image_path)
        json = jsonify(classification_results)
        status = 200
    except:
        err = traceback.format_exc()
        json = '{ "exception": "Server endpoint not responding! Please try again later." }'
        status = 500
        logging.error(err)
    resp = Response(json, status=status, mimetype='application/json')
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)