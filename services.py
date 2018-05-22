import config_accessor as cfg

import traceback
import os
import time
from requests import ReadTimeout, ConnectTimeout, HTTPError, Timeout, ConnectionError

import cv2
import numpy as np

from keras_retinanet.preprocessing.url_generator import UrlGenerator
from elasticsearch_dsl import Search, Q

import core
from models import Response, Result, Caption, AssetMeta, Suggestion, Frame
from models_es import EsAsset, EsCropped, EsAssetMeta

import logging
logger = logging.getLogger('celum.services')


# Inserts an asset into the database
def index_original_image(img, asset):
    # save original image
    original_dir = cfg.resolve(cfg.CLASSIFICATION, cfg.original_images_path)
    if not os.path.exists(original_dir):
        os.makedirs(original_dir)
        logger.info('Created new dir: {}'.format(original_dir))
    ori_file_name = '{}/{}.png'.format(original_dir, asset.asset_id)
    cv2.imwrite(ori_file_name, img)

    # insert asset into database
    logger.info('Creating validation generator...')
    es_asset = EsAsset(meta={'id': asset.asset_id},
                       asset_id=asset.asset_id,
                       asset_url=asset.url,
                       path=ori_file_name)
    es_asset.save()
    return ori_file_name


# Inserts the cropped image data of an asset into the database
def index_cropped_image(asset, img, label_name, idx):
    # save cropped image
    extraction_dir = '{}/{}'.format(cfg.resolve(cfg.CLASSIFICATION, cfg.extracted_images_path), label_name)
    if not os.path.exists(extraction_dir):
        os.makedirs(extraction_dir)
        logger.info('Created new dir: {}'.format(extraction_dir))
    cropped_file_name = '{}/{}-{}.png'.format(extraction_dir, asset.asset_id, idx)
    logger.info('Extracted image: {}'.format(cropped_file_name))
    converted_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(cropped_file_name, converted_img)
    # insert cropped image into database
    es_cropped = EsCropped(meta={'id': '{}-{}'.format(asset.asset_id, idx)},
                           asset_id=asset.asset_id,
                           parent_url=asset.url,
                           path=cropped_file_name)
    es_cropped.save()
    return cropped_file_name


# Inserts the meta data of an asset into the database
def index_asset_meta(asset, idx, caption, feature, faiss_idx):
    # store cropped image in database
    es_meta = EsAssetMeta(meta={'id': '{}-{}'.format(asset.asset_id, idx)},
                          asset_id=asset.asset_id,
                          cropped_id=idx,
                          faiss_idx=faiss_idx,
                          label=caption.label,
                          score=caption.score,
                          top_left=caption.top_left,
                          bottom_right=caption.bottom_right,
                          feature=feature)
    es_meta.save()


# Extracts features from a given image file
def extract_features(file_name):
    return core.predict_features(file_name)


# Maps the similarity index ids to asset ids
def map_index_ids_to_asset_metas(indices_ids):
    num_entries = np.array(indices_ids).shape[0]
    asset_metas = []
    search = Search(index=cfg.resolve(cfg.ELASTICSEARCH_SERVER, cfg.index_prefix) +
                          cfg.resolve(cfg.ELASTICSEARCH_SERVER, cfg.index_asset_meta))
    search.query = Q('terms', faiss_idx=indices_ids)
    search = search[:num_entries]
    response = search.execute()
    for hit in response:
        asset_metas.append(AssetMeta(hit.asset_id, hit.cropped_id, hit.faiss_idx))
    return asset_metas if response.hits.total > 0 else []


def fetch_asset_url(asset_id):
    asset = EsAsset.get(id=asset_id)
    return asset.asset_url if asset else None


def fetch_cropped_url(asset_id, cropped_id):
    cropped = EsCropped.get(id='{}-{}'.format(asset_id, cropped_id))
    return (cropped.parent_url, cropped.path) if cropped else None


# Returns a list of similar assets given a feature
def get_similar_asset_metas(feature, n=1):
    _, indices = core.index.search(feature, n + 1)
    indices = [item for item in indices[0].tolist() if item >= 0]
    asset_metas = map_index_ids_to_asset_metas(indices)
    return asset_metas


def handle_suggestion_response(result, current_asset_id, asset_metas):
    for asset_meta in asset_metas:
        asset_id = asset_meta.asset_id
        # skip if it is the same id as the classified image
        if current_asset_id == asset_id:
            continue
        cropped_id = asset_meta.cropped_id
        parent_url, path = fetch_cropped_url(asset_id, cropped_id)
        if asset_id not in result.suggestions:
            result.suggestions[asset_id] = Suggestion(parent_url,
                                                      [Frame(cropped_id, asset_meta.faiss_idx, path)])
        else:
            contains = False
            for frame in result.suggestions[asset_id].frames:
                if frame.frame_id == cropped_id:
                    contains = True
                    break
            if not contains:
                result.suggestions[asset_id].frames += [Frame(cropped_id, asset_meta.faiss_idx, path)]


def classify_content(content):
    # create a generator for fetching data
    urls = []
    for asset in content.assets:
        urls.append(asset.url)
    # prepare images for download
    val_generator = UrlGenerator(urls,
                                 cfg.resolve(cfg.RETINANET_MODEL, cfg.classes_file),
                                 cfg.resolve(cfg.RETINANET_MODEL, cfg.labels_file))

    response = Response()
    # load image
    for i, asset in enumerate(content.assets):
        logger.info('Running classification on: {}'.format(asset.url))
        # initialize result object
        result = Result()
        result.url = asset.url
        result.asset_id = asset.asset_id

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
        _, _, detections = core.model.predict_on_batch(np.expand_dims(image, axis=0))
        elapsed = time.time() - start
        logger.info('Processing time: {}'.format(elapsed))
        result.time = str(elapsed)

        # compute predicted labels and scores
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

        # correct for image scale
        detections[0, :, :4] /= scale

        # process and save detections
        for idx, (label_id, score) in enumerate(zip(predicted_labels, scores)):
            if score < cfg.resolve_float(cfg.CLASSIFICATION, cfg.min_confidence):
                continue
            # get position data
            box = detections[0, idx, :4].astype(int)
            label_name = val_generator.label_to_name(label_id)
            # save meta-info for REST API response
            caption = Caption(str(label_id),
                              label_name,
                              str(score),
                              '{};{}'.format(box[0], box[1]),   # x1;y1
                              '{};{}'.format(box[2], box[3]))   # x2;y2
            result.captions.append(caption)
            # Crop image for extraction
            h = box[3] - box[1]
            w = box[2] - box[0]
            cropped_img = draw[box[1]:(box[1] + h), box[0]:(box[0] + w)]
            # process cropped image fragment for searching
            cropped_file_name = index_cropped_image(asset, cropped_img, label_name, idx)
            features = extract_features(cropped_file_name)
            faiss_features = features.reshape((1, cfg.resolve_int(cfg.FAISS_SETTINGS, cfg.index_size)))
            # add feature to faiss index
            core.index.add(faiss_features)
            # index caption
            index_asset_meta(asset, idx, caption, features.tolist(), core.index.ntotal - 1)
            # find similar suggestions and handle response
            asset_metas = get_similar_asset_metas(faiss_features)
            handle_suggestion_response(result, asset.asset_id, asset_metas)

        # add result to response list
        response.result_list.append(result)
    return response
