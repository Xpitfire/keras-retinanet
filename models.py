class Content(object):
    def __init__(self, data):
        self.assets = [Asset(item) for item in data['assets']]

    def __repr__(self):
        return 'Content()'

    def __str__(self):
        return ''.join(['{}\n'.format(str(a)) for a in self.assets])


class Asset(object):
    def __init__(self, data):
        self.asset_id = data['asset-id']
        self.url = data['url']

    def __repr__(self):
        return 'Asset()'

    def __str__(self):
        return 'asset-id: {}, url: {}'.format(self.asset_id, self.url)


class AssetMeta(object):
    def __init__(self, asset_id, cropped_id, faiss_idx):
        self.asset_id = asset_id
        self.cropped_id = cropped_id
        self.faiss_idx = faiss_idx

    def __repr__(self):
        return 'AssetMeta()'

    def __str__(self):
        return 'asset-id: {}, cropped-id: {}, faiss-idx: {}'.format(
            self.asset_id, self.cropped_id, self.faiss_idx
        )


class Response(object):
    def __init__(self):
        self.result_list = []

    def __repr__(self):
        return 'Response()'

    def __str__(self):
        return ''.join(['{}\n'.format(str(r)) for r in self.result_list])


class Result(object):
    def __init__(self):
        self.url = None
        self.asset_id = None
        self.time = None
        self.captions = []
        self.suggestions = {}

    def __repr__(self):
        return 'Result()'

    def __str__(self):
        res = 'asset-id: {}, url: {}, time: {}\n'.format(self.asset_id, self.url, self.time)
        res += ''.join(['\tcaption: {}\n'.format(str(c)) for c in self.captions])
        res += ''.join(['\tsuggestion: {}: {}\n'.format(key, str(value)) for key, value in self.suggestions.items()])
        return res


class Caption(object):
    def __init__(self, caption_id, label, score, top_left, bottom_right):
        self.caption_id = caption_id
        self.label = label
        self.score = score
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __repr__(self):
        return 'Caption()'

    def __str__(self):
        return 'caption-id: {}, label: {}, score: {}, top-left: {}, bottom-right: {}'.format(
            self.caption_id, self.label, self.score, self.top_left, self.bottom_right
        )


class Suggestion(object):
    def __init__(self, url, frames):
        self.url = url
        self.frames = frames

    def __repr__(self):
        return 'Suggestion()'

    def __str__(self):
        res = 'url: {}\n'.format(self.url)
        res += ''.join(['\t\tframe: {}\n'.format(str(f)) for f in self.frames])
        return res


class Frame(object):
    def __init__(self, frame_id, faiss_idx, url):
        self.frame_id = frame_id
        self.faiss_idx = faiss_idx
        self.url = url

    def __repr__(self):
        return 'Frame()'

    def __str__(self):
        return 'frame-id: {}, faiss-idx: {}, url: {}'.format(
            self.frame_id, self.faiss_idx, self.url
        )
