"""
Copyright 2017-2018 Xpitfire (https://dinu.at)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import random
import threading
import time
import warnings

import keras

import json
import requests
from PIL import Image
from io import BytesIO

from .generator import Generator

from ..utils.anchors import anchor_targets_bbox
from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from ..utils.transform import transform_aabb

class UrlGenerator(Generator):
    def __init__(
        self,
        urls,
        classes_path,
        labels_path,
        **kwargs
    ):
        self.urls = urls
        self.classes_path = classes_path
        self.labels_path = labels_path
        self.load_meta_info()
        self.label_to_name(0)

        super(UrlGenerator, self).__init__(**kwargs)

    def read_image_bgr(self, image_index):
        response = requests.get(self.urls[image_index])
        image = Image.open(BytesIO(response.content))
        image = np.asarray(image.convert('RGB'))
        return image[:, :, ::-1].copy()

    def size(self):
        return len(self.urls)

    def name_to_label(self, name):
        return int(self.classes_to_labels[name])

    def label_to_name(self, label):
        return self.labels_to_classes[str(label)]

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        response = requests.get(self.urls[image_index])
        image = Image.open(BytesIO(response.content))
        return float(image.width) / float(image.height)

    def load_meta_info(self):
        with open(self.classes_path, "r") as text_file:
            self.classes_to_labels = json.load(text_file)
        with open(self.labels_path, "r") as text_file:
            self.labels_to_classes = json.load(text_file)

    def save_meta_info(self):
        with open("/tmp/classes.json", "w") as text_file:
            print(json.dumps(self.classes), file=text_file)
        with open("/tmp/labels.json", "w") as text_file:
            print(json.dumps(self.labels), file=text_file)

    def load_image(self, image_index):
        return read_image_bgr(self.image_path(image_index))