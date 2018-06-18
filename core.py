import config_accessor as cfg
import os
import csv
from shutil import copyfile
from pathlib import Path
import schedule
import time
from threading import Thread, Lock

from elasticsearch_dsl import Index
from elasticsearch_dsl.connections import connections
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model

import numpy as np

import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
import tensorflow as tf
from models_es import EsAsset, EsAssetMeta, EsCropped
import faiss

import logging

logger = logging.getLogger('celum.settings')

index = None
db_asset = None
db_asset_meta = None
db_cropped = None
model = None
feature_model = None
blacklist = []
blacklist_mutex = None
cron_job_thread = None
round_robin_backup_index = 0


def initialize_logging():
    print('Initializing logging...')
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=cfg.resolve(cfg.DEFAULT, cfg.log_dir)+cfg.resolve(cfg.DEFAULT, cfg.log_name),
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


def initialize_similarity_index():
    global index
    path = cfg.resolve(cfg.FAISS_SETTINGS, cfg.index_path)
    if not os.path.exists(path):
        os.mkdir(path)

    file = os.path.join(path, cfg.resolve(cfg.FAISS_SETTINGS, cfg.index_file))
    if not os.path.exists(file):
        index = faiss.IndexFlatIP(cfg.resolve_int(cfg.FAISS_SETTINGS, cfg.index_size))
        persist_similarity_index()
    else:
        try:
            index = faiss.read_index(file)
            logger.info("Faiss index loaded")
        except (OSError, TypeError, NameError):
            index = faiss.read_index(file)
            logger.error("Can't load index! Using default empty index")


def persist_similarity_index():
    if index is not None:
        file = os.path.join(cfg.resolve(cfg.FAISS_SETTINGS, cfg.index_path),
                            cfg.resolve(cfg.FAISS_SETTINGS, cfg.index_file))
        faiss.write_index(index, file)
        logger.info("Faiss index saved to disk")
    else:
        logger.warning("Can't save, index was not loaded yet!")


def initialize_elastic_search():
    global db_asset, db_cropped, db_asset_meta
    connections.create_connection(hosts=cfg.resolve(cfg.ELASTICSEARCH_SERVER, cfg.host),
                                  port=cfg.resolve(cfg.ELASTICSEARCH_SERVER, cfg.port),
                                  timeout=20)

    db_asset = Index(cfg.resolve(cfg.ELASTICSEARCH_SERVER,
                                 cfg.index_prefix) + cfg.resolve(cfg.ELASTICSEARCH_SERVER, cfg.index_asset))
    if not db_asset.exists():
        db_asset.doc_type(EsAsset)
        db_asset.create()
    db_asset_meta = Index(cfg.resolve(cfg.ELASTICSEARCH_SERVER,
                                      cfg.index_prefix) + cfg.resolve(cfg.ELASTICSEARCH_SERVER, cfg.index_asset_meta))
    if not db_asset_meta.exists():
        db_asset_meta.doc_type(EsAssetMeta)
        db_asset_meta.create()
    db_cropped = Index(cfg.resolve(cfg.ELASTICSEARCH_SERVER,
                                   cfg.index_prefix) + cfg.resolve(cfg.ELASTICSEARCH_SERVER, cfg.index_cropped))
    if not db_cropped.exists():
        db_cropped.doc_type(EsCropped)
        db_cropped.create()

    logger.info("Elastic search initialized!")


def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    return tf.Session(config=cfg)


def initialize_retinanet():
    global model
    logger.info('Loading retinanet classification model...')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logger.info('Setting keras session...')
    keras.backend.tensorflow_backend.set_session(get_session())

    logger.info('Loading model name...')
    model = keras.models.load_model(cfg.resolve(cfg.RETINANET_MODEL, cfg.model_path) +
                                    cfg.resolve(cfg.RETINANET_MODEL, cfg.model_name),
                                    custom_objects=custom_objects)


def initialize_extraction_model():
    global feature_model
    logger.info('Loading extraction model...')
    resnet = ResNet50(weights='imagenet')
    output = resnet.layers[-2].output
    feature_model = Model(resnet.input, output)


def initialize_blacklist():
    global blacklist, blacklist_mutex
    blacklist_mutex = Lock()

    path = cfg.resolve(cfg.FAISS_SETTINGS, cfg.index_path)
    if not os.path.exists(path):
        os.mkdir(path)
    file = os.path.join(path, cfg.resolve(cfg.FAISS_SETTINGS, cfg.index_blacklist_file))
    if not os.path.exists(file):
        Path(file).touch()
    with open(file,'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            blacklist.append(int(row['index']))


# Multi-threading is currently not used/required - preventive method!
def threadsafe_blacklist_operation(delegate):
    """
    Thread safe implementation for handling multiple blacklist operations at once.
    :param delegate:
    :return:
    """
    blacklist_mutex.acquire()
    try:
        return delegate(blacklist)
    finally:
        blacklist_mutex.release()


def persist_blacklist_index():
    path = cfg.resolve(cfg.FAISS_SETTINGS, cfg.index_path)
    file = os.path.join(path, cfg.resolve(cfg.FAISS_SETTINGS,
                                          cfg.index_blacklist_file))
    with open(file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['index'])
        writer.writeheader()
        blacklist_dict = threadsafe_blacklist_operation(lambda bl: [{'index': id} for id in bl])
        writer.writerows(blacklist_dict)


def predict_features(img_file):
    x = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    return feature_model.predict(x)[0]


def backup_persisting_files():
    global round_robin_backup_index
    round_robin_backup_index += 1
    round_robin_backup_index %= cfg.resolve_int(cfg.CRON_JOB, cfg.cron_job_round_robin_backups)

    # copy faiss file
    path = cfg.resolve(cfg.FAISS_SETTINGS, cfg.index_path)
    file = os.path.join(path, cfg.resolve(cfg.FAISS_SETTINGS, cfg.index_file))
    copyfile(file, file+'.backup_{}'.format(round_robin_backup_index))

    # copy blacklist file
    file = os.path.join(path, cfg.resolve(cfg.FAISS_SETTINGS, cfg.index_blacklist_file))
    copyfile(file, file+'.backup_{}'.format(round_robin_backup_index))


def trigger_backup():
    logging.info('Backup persisting files...')
    backup_persisting_files()
    logger.info('Persisting blacklist...')
    persist_blacklist_index()
    logger.info('Persisting faiss index...')
    persist_similarity_index()


def initialize_cron_job():
    global cron_job_thread

    def cron_job_runner():
        schedule.every(cfg.resolve_int(cfg.CRON_JOB, cfg.cron_job_interval)).minutes.do(trigger_backup)
        while True:
            schedule.run_pending()
            time.sleep(1)

    if not cron_job_thread:
        cron_job_thread = Thread(target=cron_job_runner)
        cron_job_thread.start()
