import logging
import traceback
import os
import time
from requests import ReadTimeout, ConnectTimeout, HTTPError, Timeout, ConnectionError

import cv2
import numpy as np

from keras_retinanet.preprocessing.url_generator import UrlGenerator

import settings

log = logging.getLogger('celum.classify')


def index_original_image(img):

    pass


def index_copped_image(img, label_name, idx):
    # cropped = image[box[2]:box[3], box[0]:box[1]]
    ts = time.time()
    extraction_dir = 'data/extracted/{}'.format(label_name)
    if not os.path.exists(extraction_dir):
        os.makedirs(extraction_dir)
        log.info('Created new dir: {}'.format(extraction_dir))
    cropped_file_name = '{}/{}_{}.png'.format(extraction_dir, ts, idx)
    log.info('Extracted image: {}'.format(cropped_file_name))
    converted_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(cropped_file_name, converted_img)


def classify_content(content):
    # create a generator for testing data
    log.info('Creating validation generator...')

    urls = []
    for asset in content['assets']:
        urls.append(asset['url'])
    # prepare images for download
    val_generator = UrlGenerator(urls,
                                 settings.config['RETINANET_MODEL']['classes_file'],
                                 settings.config['RETINANET_MODEL']['labels_file'])
    result_list = []
    # load image
    for i, asset in enumerate(content['assets']):
        log.info('Running classification on: {}'.format(asset['url']))
        # initialize result object
        result = {
            'url': asset['url'],
            'asset-id': asset['asset-id']
        }
        log.info('Reading image bgr...')
        try:
            # fetch images
            image = val_generator.read_image_bgr(i)
            # index original image for searching
            index_original_image(image)
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

        # classify image
        start = time.time()
        _, _, detections = settings.model.predict_on_batch(np.expand_dims(image, axis=0))
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
        for idx, (label_id, score) in enumerate(zip(predicted_labels, scores)):
            if score < 0.5:
                continue
            # get position data
            box = detections[0, idx, :4].astype(int)
            # Crop image for extraction
            h = box[3] - box[1]
            w = box[2] - box[0]
            cropped = draw[box[1]:(box[1] + h), box[0]:(box[0] + w)]
            label_name = val_generator.label_to_name(label_id)

            # process cropped image fragment for searching
            index_copped_image(cropped, label_name, idx)

            # save meta-info for REST API response
            caption = {'id': str(label_id),
                       'label': label_name,
                       'score': str(score),
                       'top-left': '{};{}'.format(box[0], box[1]),         # x1;y1
                       'bottom-right': '{};{}'.format(box[2], box[3])}     # x2;y2
            captions.append(caption)

        result['captions'] = captions
        result_list.append(result)

    return {'results': result_list}
