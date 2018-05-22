import configparser


DEFAULT = 'DEFAULT'
CLASSIFICATION = 'CLASSIFICATION'
ELASTICSEARCH_SERVER = 'ELASTICSEARCH_SERVER'
RETINANET_SERVER = 'RETINANET_SERVER'
RETINANET_MODEL = 'RETINANET_MODEL'
FAISS_SETTINGS = 'FAISS_SETTINGS'

log_dir = 'log_dir'
log_name = 'log_name'
host = 'host'
port = 'port'

index_path = 'index_path'
index_file = 'index_file'
index_size = 'index_size'
index_prefix = 'index_prefix'
index_asset = 'index_asset'
index_asset_meta = 'index_asset_meta'
index_cropped = 'index_cropped'

model_path = 'model_path'
model_name = 'model_name'
classes_file = 'classes_file'
labels_file = 'labels_file'

min_confidence = 'min_confidence'
original_images_path = 'original_images_path'
extracted_images_path = 'extracted_images_path'

print('Reading configurations...')
config = configparser.ConfigParser()
config.read('retinanet.cfg')


def resolve(section, prop):
    return config[section][prop]


def resolve_int(section, prop):
    return int(resolve(section, prop))


def resolve_float(section, prop):
    return float(resolve(section, prop))
