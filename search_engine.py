from elasticsearch_dsl import DocType, Integer, Keyword, Text
import configparser

import logging
logger = logging.getLogger('celum.search_engine')


# Define elasticsearch asset DocType -> easy persistence with elasticsearch
class ESQueryAsset(DocType):
    asset_id = Keyword()
    asset_url = Text()
    asset_encoding = []
    asset_similarity_index = Integer()

    class Meta:
        config = configparser.ConfigParser()
        config.read('retinanet.cfg')
        index = config["ELASTICSEARCH_SERVER"]["index_name"]

    def save(self, **kwargs):
        return super(ESQueryAsset, self).save(**kwargs)


def save(asset):
    es_asset = ESQueryAsset(meta={'id': asset.asset_id},
                            asset_id=asset.asset_id,
                            asset_url=asset.asset_url,
                            asset_encoding=asset.asset_encoding.tolist(),
                            asset_similarity_index=asset.similarity_index)
    es_asset.save()
