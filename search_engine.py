import configparser

import settings
import logging
logger = logging.getLogger('celum.search_engine')

config = configparser.ConfigParser()
config.read('retinanet.cfg')
search_index_prefix = config["ELASTICSEARCH_SERVER"]["index_prefix"]


def insert_auto(doc, doc_type):
    index_name = '{}_{}'.format(search_index_prefix, doc_type)
    return settings.search.index(index=index_name,
                                 doc_type=doc_type,
                                 body=doc)


def insert(asset_id, doc, doc_type):
    index_name = '{}_{}'.format(search_index_prefix, doc_type)
    return settings.search.create(index=index_name,
                                  doc_type=doc_type,
                                  id=asset_id,
                                  body=doc)


def get(asset_id, doc_type):
    index_name = '{}_{}'.format(search_index_prefix, doc_type)
    return settings.search.get(index=index_name,
                               doc_type=doc_type,
                               id=asset_id)


def update(asset_id, ref_id, doc_type):
    index_name = '{}_{}'.format(search_index_prefix, doc_type)
    doc = {
        'script': 'ctx._source.captions = "{}"'.format(ref_id)
    }
    settings.search.update(index=index_name,
                           doc_type=doc_type,
                           id=asset_id,
                           body=doc)
