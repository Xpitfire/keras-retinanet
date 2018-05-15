import configparser
from elasticsearch_dsl import DocType, Keyword, Text

import logging
logger = logging.getLogger('celum.search_engine')

config = configparser.ConfigParser()
config.read('retinanet.cfg')
search_index_prefix = config["ELASTICSEARCH_SERVER"]["index_prefix"]


class EsAsset(DocType):
    asset_id = Keyword()
    asset_url = Text()
    path = Text()

    class Meta:
        index = config["ELASTICSEARCH_SERVER"]["index_prefix"] + config["ELASTICSEARCH_SERVER"]["index_asset"]

    def save(self, **kwargs):
        return super(EsAsset, self).save(**kwargs)


class EsAssetMeta(DocType):
    asset_id = Keyword()
    cropped_id = Keyword()
    faiss_idx = Text()
    label = Text()
    score = Text()
    top_left = Text()
    bottom_right = Text()
    feature = Text()

    class Meta:
        index = config["ELASTICSEARCH_SERVER"]["index_prefix"] + config["ELASTICSEARCH_SERVER"]["index_asset_meta"]

    def save(self, **kwargs):
        return super(EsAssetMeta, self).save(**kwargs)


class EsCropped(DocType):
    asset_id = Keyword()
    parent_url = Text()
    path = Text()

    class Meta:
        index = config["ELASTICSEARCH_SERVER"]["index_prefix"] + config["ELASTICSEARCH_SERVER"]["index_cropped"]

    def save(self, **kwargs):
        return super(EsCropped, self).save(**kwargs)

