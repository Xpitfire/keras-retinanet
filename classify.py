import logging
import traceback
import os
import time
from requests import ReadTimeout, ConnectTimeout, HTTPError, Timeout, ConnectionError

import cv2
import numpy as np

import keras
import keras.preprocessing.image
import tensorflow as tf

from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.preprocessing.url_generator import UrlGenerator

import settings

is_model_loaded = False
model = None
generator = None

log = logging.getLogger('celum.classify')


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
        log.info('Model not loaded...')
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        log.info('Setting keras session...')
        keras.backend.tensorflow_backend.set_session(get_session())

        log.info('Loading model name...')
        model = keras.models.load_model(settings.config['RETINANET_MODEL']['model_path']+settings.config['RETINANET_MODEL']['model_name'],
                                        custom_objects=custom_objects)

        log.info('Creating image data generator...')
        generator = keras.preprocessing.image.ImageDataGenerator()
        is_model_loaded = True
    else:
        log.info('Model already loaded...')


def classify_urls(urls):
    # init model
    init_classification()

    # create a generator for testing data
    log.info('Creating validation generator...')
    val_generator = UrlGenerator(urls,
                                 settings.config['RETINANET_MODEL']['classes_file'],
                                 settings.config['RETINANET_MODEL']['labels_file'])

    result_list = []
    # load image
    for i in range(len(urls)):
        log.info('Running classification on: {}'.format(urls[i]))

        result = {'url': urls[i]}

        log.info('Reading image bgr...')
        try:
            image = val_generator.read_image_bgr(i)
        except (OSError, ConnectTimeout, HTTPError, ReadTimeout, Timeout, ConnectionError):
            log.warning('Skipped: Unable to reach resource')
            continue
        except:
            err = traceback.format_exc()
            log.error('Could not read image: {}'.format(err))
            continue

        # copy to draw on
        log.info('Drawing cvt color...')
        draw = np.asarray(image.copy())
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # pre-process the image for the network
        log.info('Processing image...')
        image = val_generator.preprocess_image(image)
        image, scale = val_generator.resize_image(image)

        # process image
        start = time.time()
        _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
        elapsed = time.time() - start
        log.info('Processing time: {}'.format(elapsed))
        result['time'] = str(elapsed)

        # compute predicted labels and scores
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

        # correct for image scale
        detections[0, :, :4] /= scale

        captions = []
        # process and save detections
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score < 0.5:
                continue
            # get position data
            box = detections[0, idx, :4].astype(int)
            # Crop image for extraction
            h = box[3] - box[1]
            w = box[2] - box[0]
            cropped = draw[box[1]:(box[1] + h), box[0]:(box[0] + w)]

            # cropped = image[box[2]:box[3], box[0]:box[1]]
            label_name = val_generator.label_to_name(label)
            ts = time.time()
            extraction_dir = 'data/extracted/{}'.format(label_name)
            if not os.path.exists(extraction_dir):
                os.makedirs(extraction_dir)
                log.info('Created new dir: {}'.format(extraction_dir))
            cropped_file_name = '{}/{}_{}.png'.format(extraction_dir, ts, idx)
            log.info('Extracted image: {}'.format(cropped_file_name))
            cv2.imwrite(cropped_file_name, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

            # save meta-info for REST API response
            caption = {'id': str(label),
                       'label': label_name,
                       'score': str(score),
                       'top-left': '{};{}'.format(box[0], box[1]),         # x1;y1
                       'bottom-right': '{};{}'.format(box[2], box[3])}     # x2;y2
            captions.append(caption)

        result['captions'] = captions
        result_list.append(result)

    return {'results': result_list}
