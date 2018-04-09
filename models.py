import json


class Result(object):
    def __init__(self, json_str=None):
        self.time_elapsed = None
        self.url = None
        self.detections = None
        self.caption_list = []
        if json_str and isinstance(json_str, str):
            self.__dict__ = json.loads(json_str)
