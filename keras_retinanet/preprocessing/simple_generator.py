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

import json

from .generator import Generator


class SimpleGenerator(Generator):
    """Class to load images according to the provided url request string.
    """

    def __init__(
            self,
            classes_path,
            labels_path,
            **kwargs
    ):
        self.classes_path = classes_path
        self.labels_path = labels_path
        self.load_meta_info()
        self.label_to_name(0)

        super(SimpleGenerator, self).__init__(**kwargs)

    def read_image_bgr(self, image_index):
        pass

    def size(self):
        return 0

    def name_to_label(self, name):
        return int(self.classes_to_labels[name])

    def label_to_name(self, label):
        return self.labels_to_classes[str(label)]

    def image_aspect_ratio(self, image_index):
        return 1.0

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
        return self.read_image_bgr(self.image_path(image_index))
