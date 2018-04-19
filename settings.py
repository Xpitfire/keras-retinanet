import configparser
import logging
import faiss
import os
from elasticsearch import Elasticsearch

config = None
index = None
search = None


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
            logging.info("Faiss index loaded")
        except (OSError, TypeError, NameError):
            index = faiss.read_index(file)
            logging.error("Can't load index! Using default empty index")


def persist_similarity_index():
    if index is not None:
        faiss.write_index(index,
                          config["FAISS_SETTINGS"]["index_path"] +
                          config["FAISS_SETTINGS"]["index_file"])
        logging.info("Faiss index saved to disk")
    else:
        logging.warning("Can't save, index was not loaded yet!")


def initialize_elastic_search():
    global search
    search = Elasticsearch(hosts=[{'host': config["ELASTICSEARCH_SERVER"]["host"],
                                   'port': config["ELASTICSEARCH_SERVER"]["port"]}])
    logging.info("Elastic search initialized!")
