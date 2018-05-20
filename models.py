class Content(object):
    def __init__(self, data):
        self.assets = [Asset(item) for item in data['assets']]


class Asset(object):
    def __init__(self, data):
        self.asset_id = data['asset-id']
        self.url = data['url']


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
    def __init__(self, id, label, score, top_left, bottom_right):
        self.id = id
        self.label = label
        self.score = score
        self.top_left = top_left
        self.bottom_right = bottom_right


class SimilarSuggestions(object):
    def __init__(self, data):
        self.data = data

    def __getattr__(self,key):
        return [Suggestion(item) for item in self.data[key]]


class Suggestion(object):
    def __init__(self, data):
        self.url = None
        self.frames = []


class Frame(object):
    def __init__(self):
        self.frame_id = None
        self.faiss_idx = None
        self.url = None