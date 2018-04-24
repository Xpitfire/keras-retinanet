from settings import index, config
import faiss
import logging
logger = logging.getLogger('celum.indexer')


def persist():
    if index is not None:
        faiss.write_index(index,
                          config["FAISS_SETTINGS"]["index_path"] +
                          config["FAISS_SETTINGS"]["index_file"])
        logger.info("Faiss index saved to disk")
    else:
        logger.warning("Can't save, index was not loaded yet!")
