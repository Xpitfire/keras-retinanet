from tensorflow.python.keras.preprocessing import image

import numpy as np
import settings


def predict(img_file):
    x = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    return settings.extraction_model.predict(x)[0]


