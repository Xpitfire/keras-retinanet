import redis
from flask import Flask
from flask import request
from flask import Response

import sys
import cv2
import os
import time
import numpy as np

import keras
import keras.preprocessing.image
import tensorflow as tf

from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.preprocessing.url_generator import UrlGenerator

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

# model properties
classes_path = './data/coco/classes.json'
labels_path = './data/coco/labels.json'
model_name = 'snapshots/resnet50_coco_best.h5'
is_model_loaded = False
model = None
generator = None

class Result(object):
    def __init__(self):
        self.time_elapsed = None
        self.url = None
        self.detections = None
        self.caption_list = []

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def init_classification():
    global is_model_loaded
    global model
    global generator

    if not is_model_loaded:
        print('Model not loaded...')
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Setting keras session...')
        keras.backend.tensorflow_backend.set_session(get_session())

        print('Loading model name...')
        model = keras.models.load_model(model_name, custom_objects=custom_objects)

        print('Creating image data generator...')
        generator = keras.preprocessing.image.ImageDataGenerator()
        is_model_loaded = True
    else:
        print('Model already loaded...')

def classify_urls(urls):
    init_classification()

    # create a generator for testing data
    print('Creating validation generator...')
    val_generator = UrlGenerator(urls, classes_path, labels_path)

    results = []

    # load image
    for i in range(len(urls)):
        print('Running classification on', urls[i])

        result = Result()
        result.url = urls[i]

        print('Reading image bgr...')
        image = val_generator.read_image_bgr(i)

        # copy to draw on
        print('Drawing cvt color...')
        draw = np.asarray(image.copy())
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        print('Processing image...')
        image = val_generator.preprocess_image(image)
        image, scale = val_generator.resize_image(image)

        # process image
        start = time.time()
        _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
        elapsed = time.time() - start
        print("Processing time: ", elapsed)
        result.time_elapsed = elapsed

        # compute predicted labels and scores
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

        # correct for image scale
        detections[0, :, :4] /= scale

        # visualize detections
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score < 0.5:
                continue
            b = detections[0, idx, :4].astype(int)
            cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
            caption = "{} {:.3f}".format(val_generator.label_to_name(label), score)
            result.caption_list.append((label, val_generator.label_to_name(label), score, b))

        results.append(result)
    
    return results

def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

def jsonify(classification_results):
    json = '{ "results": [ '
    for i, result in enumerate(classification_results):
        json += '{ '
        # time
        json += '"time": '
        json += '"'
        json += str(result.time_elapsed)
        json += '", '
        # url
        json += '"url": '
        json += '"'
        json += result.url
        json += '", '
        # captions
        json += '"captions": [ '
        for j, item in enumerate(result.caption_list):
            id, label, score, box = item
            json += '{ '
            # id
            json += '"id": '
            json += '"'
            json += str(id)
            json += '", '
            # label
            json += '"label": '
            json += '"'
            json += label
            json += '", '
            # score
            json += '"score": '
            json += '"'
            json += str(score)
            json += '", '
            # box
            json += '"top-left": '
            json += '"'
            json += str(box[0]) + ';' + str(box[1]) # (x1, y1)
            json += '", '
            json += '"bottom-right": '
            json += '"'
            json += str(box[2]) + ';' + str(box[3]) # (x2, y2)
            json += '"'
            json += ' }'
            if j < len(result.caption_list)-1:
                json += ', '
        json += ' ] '
        json += ' }'
        if i < len(classification_results)-1:
            json += ', '
    json += ' ] }'
    return json

@app.route('/')
def home():
    return 'Hello RetinaNet!'

@app.route('/count')
def count():
    count = get_hit_count()
    return 'Demo: I have been seen {} times.\n'.format(count)

@app.route("/classify")
def classify():
    try:
        image_path = request.args.get('url').split(';')
        print(image_path)
        classification_results = classify_urls(image_path)
        json = jsonify(classification_results)
        status = 200
    except:
        json = '{ "exception": "' + str(sys.exc_info()[0]) + '" }'
        status = 500
    resp = Response(json, status=status, mimetype='application/json')
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)