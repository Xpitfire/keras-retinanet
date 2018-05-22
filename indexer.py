import config_accessor as cfg

from settings import index
import faiss

import logging
logger = logging.getLogger('celum.indexer')


def persist():
    if index is not None:
        faiss.write_index(index,
                          cfg.resolve(cfg.FAISS_SETTINGS, cfg.index_path) +
                          cfg.resolve(cfg.FAISS_SETTINGS, cfg.index_file))
        logger.info("Faiss index saved to disk")
    else:
        logger.warning("Can't save, index was not loaded yet!")
