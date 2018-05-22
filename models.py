class Content(object):
    def __init__(self, data):
        self.assets = [Asset(item) for item in data['assets']]


class Asset(object):
    def __init__(self, data):
        self.asset_id = data['asset-id']
        self.url = data['url']


class AssetMeta(object):
    def __init__(self, asset_id, cropped_id, faiss_idx):
        self.asset_id = asset_id
        self.cropped_id = cropped_id
        self.faiss_idx = faiss_idx


class Response(object):
    def __init__(self):
        self.result_list = []


class Result(object):
    def __init__(self):
        self.url = None
        self.asset_id = None
        self.time = None
        self.captions = []
        self.suggestions = {}


class Caption(object):
    def __init__(self, caption_id, label, score, top_left, bottom_right):
        self.caption_id = caption_id
        self.label = label
        self.score = score
        self.top_left = top_left
        self.bottom_right = bottom_right


class Suggestion(object):
    def __init__(self, url, frames):
        self.url = url
        self.frames = frames


class Frame(object):
    def __init__(self, frame_id, faiss_idx, url):
        self.frame_id = frame_id
        self.faiss_idx = faiss_idx
        self.url = url
