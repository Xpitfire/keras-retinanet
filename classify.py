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

from models import Result

# model properties
classes_path = './data/coco/classes.json'
labels_path = './data/coco/labels.json'
model_name = 'snapshots/resnet50_coco_best_v2.0.1.h5'

is_model_loaded = False
model = None
generator = None

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def init_classification():
    global is_model_loaded
    global model
    global generator

    # if model not already loaded, load new model
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
    # init model
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

        # process and save detections
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score < 0.5:
                continue
            # get position data
            b = detections[0, idx, :4].astype(int)
            # Crop image for extraction
            h = b[3] - b[1]
            w = b[2] - b[0]
            cropped = draw[b[1]:(b[1]+h), b[0]:(b[0]+w)]

            #cropped = image[b[2]:b[3], b[0]:b[1]]
            label_name = val_generator.label_to_name(label)
            ts = time.time()
            extraction_dir = "data/extracted/{}".format(label_name)
            if not os.path.exists(extraction_dir):
                os.makedirs(extraction_dir)
                print("Created new dir: ", extraction_dir)
            cropped_file_name = "{}/{}_{}.png".format(extraction_dir, ts, idx)
            print("Extracted image: ", cropped_file_name)
            cv2.imwrite(cropped_file_name, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

            # save meta-info for REST API response
            result.caption_list.append((label, label_name, score, b))

        results.append(result)
    
    return results