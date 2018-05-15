import traceback
import os
import time
from requests import ReadTimeout, ConnectTimeout, HTTPError, Timeout, ConnectionError

import cv2
import numpy as np

from keras_retinanet.preprocessing.url_generator import UrlGenerator

import settings
from search_engine import EsAsset, EsCropped, EsAssetMeta
import feature_extractor
import indexer
import faiss

import logging
logger = logging.getLogger('celum.services')


def index_original_image(img, asset):
    # save original image
    original_dir = 'data/original'
    if not os.path.exists(original_dir):
        os.makedirs(original_dir)
        logger.info('Created new dir: {}'.format(original_dir))
    ori_file_name = '{}/{}.png'.format(original_dir, asset['asset-id'])
    cv2.imwrite(ori_file_name, img)

    # insert asset into database
    logger.info('Creating validation generator...')
    es_asset = EsAsset(meta={'id': asset['asset-id']},
                       asset_id=asset['asset-id'],
                       asset_url=asset['url'],
                       path=ori_file_name)
    es_asset.save()


def index_cropped_image(asset, img, label_name, idx):
    # save cropped image
    extraction_dir = 'data/extracted/{}'.format(label_name)
    if not os.path.exists(extraction_dir):
        os.makedirs(extraction_dir)
        logger.info('Created new dir: {}'.format(extraction_dir))
    cropped_file_name = '{}/{}-{}.png'.format(extraction_dir, asset['asset-id'], idx)
    logger.info('Extracted image: {}'.format(cropped_file_name))
    converted_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(cropped_file_name, converted_img)

    # extract features
    pred = feature_extractor.predict(cropped_file_name).tolist()

    # insert cropped image into database
    es_cropped = EsCropped(meta={'id': '{}-{}'.format(asset['asset-id'], idx)},
                           asset_id=asset['asset-id'],
                           path=cropped_file_name)
    es_cropped.save()
    return pred


def index_asset_meta(asset, idx, caption, feature):
    # store cropped image in database
    es_meta = EsAssetMeta(meta={'id': '{}-{}'.format(asset['asset-id'], idx)},
                          asset_id=asset['asset-id'],
                          cropped_id=idx,
                          label=caption['label'],
                          score=caption['score'],
                          top_left=caption['top-left'],
                          bottom_right=caption['bottom-right'],
                          feature=feature)
    es_meta.save()


def classify_content(content):
    # create a generator for testing data
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
        logger.info('Running classification on: {}'.format(asset['url']))
        # initialize result object
        result = {
            'url': asset['url'],
            'asset-id': asset['asset-id']
        }
        logger.info('Reading image bgr...')
        try:
            # fetch images
            image = val_generator.read_image_bgr(i)
            # index original image for searching
            index_original_image(image, asset)
        except (OSError, ConnectTimeout, HTTPError, ReadTimeout, Timeout, ConnectionError):
            logger.warning('Skipped: Unable to reach resource')
            continue
        except:
            err = traceback.format_exc()
            logger.error('Could not read image: {}'.format(err))
            continue

        # copy to draw on
        logger.info('Drawing cvt color...')
        draw = np.asarray(image.copy())
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # pre-process the image for the network
        logger.info('Processing image...')
        image = val_generator.preprocess_image(image)
        image, scale = val_generator.resize_image(image)

        # classify image
        start = time.time()
        _, _, detections = settings.model.predict_on_batch(np.expand_dims(image, axis=0))
        elapsed = time.time() - start
        logger.info('Processing time: {}'.format(elapsed))
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
            label_name = val_generator.label_to_name(label_id)
            # save meta-info for REST API response
            caption = {'id': str(label_id),
                       'label': label_name,
                       'score': str(score),
                       'top-left': '{};{}'.format(box[0], box[1]),         # x1;y1
                       'bottom-right': '{};{}'.format(box[2], box[3])}     # x2;y2
            captions.append(caption)
            # Crop image for extraction
            h = box[3] - box[1]
            w = box[2] - box[0]
            cropped_img = draw[box[1]:(box[1] + h), box[0]:(box[0] + w)]
            # process cropped image fragment for searching
            pred = index_cropped_image(asset, cropped_img, label_name, idx)
            # index caption
            index_asset_meta(asset, idx, caption, pred)

        result['captions'] = captions
        result_list.append(result)

    return {'results': result_list}
