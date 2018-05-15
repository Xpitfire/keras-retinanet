import configparser
import faiss
import os
from elasticsearch_dsl import Index
from elasticsearch_dsl.connections import connections
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.models import Model

import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
import tensorflow as tf
from search_engine import EsAsset

import logging
logger = logging.getLogger('celum.settings')


config = None
index = None
search = None
model = None
extraction_model = None


def initialize_settings():
    global config
    print('Reading configurations...')
    config = configparser.ConfigParser()
    config.read('retinanet.cfg')


def initialize_logging():
    print('Initializing logging...')
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=config['DEFAULT']['log_dir'] + config['DEFAULT']['log_name'],
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
    path = config["FAISS_SETTINGS"]["index_path"]
    if not os.path.exists(path):
        os.mkdir(path)

    file = path + config["FAISS_SETTINGS"]["index_file"]
    if not os.path.exists(file):
        index = faiss.IndexFlatL2(int(config["FAISS_SETTINGS"]["index_size"]))
    else:
        try:
            index = faiss.read_index(file)
            logger.info("Faiss index loaded")
        except (OSError, TypeError, NameError):
            index = faiss.read_index(file)
            logger.error("Can't load index! Using default empty index")


def initialize_elastic_search():
    global search
    connections.create_connection(hosts=config["ELASTICSEARCH_SERVER"]["host"],
                                  port=config["ELASTICSEARCH_SERVER"]["port"],
                                  timeout=20)

    index = Index('retinanet_demo')
    if not index.exists():
        index.doc_type(EsAsset)
        index.create()

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
    model = keras.models.load_model(config['RETINANET_MODEL']['model_path'] +
                                    config['RETINANET_MODEL']['model_name'],
                                    custom_objects=custom_objects)


def initialize_extraction_model():
    global extraction_model
    logger.info('Loading extraction model...')
    resnet = ResNet50(weights='imagenet')
    output = resnet.layers[-2].output
    extraction_model = Model(resnet.input, output)
