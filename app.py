import redis
from flask import Flask
from flask import request, redirect
from flask import Response

import traceback

from misc import jsonp, jsonify
from models import Result
from classify import classify_urls

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)



def resolve_special_url_cmd(image_path):
    result_list = []
    for path in image_path:
        if 'file://' in path and path.endswith('.list'):
            path = url.replace('file://','')
            content = [line.strip() for line in open(path, 'r').readlines()]
            result_list += content
        else:
            result_list.append(path)

@app.route("/classify", methods=['GET'])
@jsonp
def classify():
    try:
        image_path = request.args.get('url').split(';')
        print(image_path)
        resolve_special_url_cmd(image_path)
        classification_results = classify_urls(image_path)
        json = jsonify(classification_results)
        status = 200
    except:
        err = traceback.format_exc()
        json = '{ "exception": "Server endpoint not responding! Please try again later." }'
        status = 500
        print(err)
    resp = Response(json, status=status, mimetype='application/json')
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)