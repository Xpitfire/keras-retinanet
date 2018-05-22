import unittest
import core
import services
import misc
import json

from models import Content


class ServicesTestCase(unittest.TestCase):
    def setUp(self):
        core.initialize_logging()
        core.initialize_similarity_index()
        core.initialize_elastic_search()
        core.initialize_retinanet()
        core.initialize_extraction_model()

    def test_get_req_content_res(self):
        id_ = 'test'
        img_url = 'https://www.scienceabc.com/wp-content/uploads/2016/10/Plane-flying-on-earth-atmosphere.jpg'
        req = misc.classify_get_req_to_content(id_, img_url)
        res = services.classify_content(req)
        self.assertEqual(len(res.result_list), 1, 'Missing response result')
        self.assertEqual(id_, res.result_list[0].asset_id)
        self.assertEqual(img_url, res.result_list[0].url)

    def test_post_req_content_res(self):
        with open('tests/services/test_req1.json') as json_data:
            req = json.load(json_data)
            json_data.close()
            res = services.classify_content(Content(req))
            self.assertEqual(len(res.result_list), 4, 'Missing response result')

