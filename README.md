# Keras RetinaNet [![Build Status](https://travis-ci.org/fizyr/keras-retinanet.svg?branch=master)](https://travis-ci.org/fizyr/keras-retinanet)
Keras implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

## Docker Setup (the easy way...)

Pull the docker container from the docker hub by executing the following command:
```
docker run --name retinanet --runtime=nvidia -p 5000:5000 -p 9090:9090 -v /home/celum/DataSets/:/home/celum/DataSets -it xpitfire/kerasretinanet_web
```

Here we are mapping a directory directly to the docker instance. The `--runtime=nvidia` option assumes that there is GPU support via `nvidia-docker2`. To operate on CPU only remove that option.

### Using docker compose

To build after checking out the repository execute:
```
docker-compose build
```

To start the container execute:
```
docker-compose up
```

Note that to enable GPU support it is required to install and setup [nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) first. To configure docker-compose to use nvidia runtime you have to configure the default runtime of docker to use nvidia. See [here](https://github.com/NVIDIA/nvidia-docker/issues/568). It is also required to remove the `#` at the `tenworflow-gpu` line from the [requirements.txt](requirements.txt) file.

## Service Usage

Classification request can be sent to port `5000`:
```
http://localhost:5000?url=http://image.url.com/id1234.jpg
```

It is also possible to classify images from the file system by defining the protocol as `file://`:
```
http://localhost:5000?url=file:///path/to/image1234.jpg
```

Multiple images may be classified by separating them with a `;` as shown in the following example:
```
http://localhost:5000/classify?url=file:///home/celum/DataSets/voest_test_small/_75_1349031.png;file:///home/celum/DataSets/voest_test_small/_93_2000057.png
```

## Installation requirements to run RetinaNet (the hard way...)

1) Clone this repository.
2) In the repository, execute `python setup.py install --user` or use pip such as `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements. We tested `tensorflow` version 1.4.0 with `tensorflow-gpu` version 1.4.0.
   To install tensorflow 1.4.0 via anaconda enter: `conda install -c conda-forge tensorflow`.
   To install `tensorflow-gpu` 1.4.0 via anaconda enter: `conda install -c aaronzs tensorflow-gpu`.
   Also, make sure Keras 2.1.3 is installed.
   To install Keras version 2.1.3 via anaconda enter: `conda install -c anaconda keras`.
3) As of writing, this repository requires the master branch of `keras-resnet` (run `pip install --user --upgrade git+https://github.com/broadinstitute/keras-resnet`).
4) Optionally, install `pycocotools` if you want to train / test on the MS COCO dataset by running `pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`.
   If working on Windows you might require to change the following property to be able to build: `cd cocoapi\PythonAPI` edit `setup.py` by setting the property `extra_compile_args={'gcc': ['/Qstd=c99']}`.
   Furthermore, the CocoGenerator requires the usage of Pillow version 4.0.0. Install Pillow via pip: `pip install Pillow==4.0.0`

## Training
`keras-retinanet` can be trained using [this](https://github.com/Xpitfire/keras-retinanet/blob/master/keras_retinanet/bin/train.py) script.
Note that the train script uses relative imports since it is inside the `keras_retinanet` package.
If you want to adjust the script for your own use outside of this repository,
you will need to switch it to use absolute imports.

If you installed `keras-retinanet` correctly, the train script will be installed as `retinanet-train`.
However, if you make local modifications to the `keras-retinanet` repository, you should run the script directly from the repository.
That will ensure that your local changes will be used by the train script.

## Data

To get imagenet test data follow the link below to the github repository:

`https://github.com/dividiti/ck-caffe/tree/master/script/imagenet-downloader`

### Usage
For training on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/), run:
```
# Running directly from the repository:
keras_retinanet/bin/train.py pascal <path to VOCdevkit/VOC2007>

# Using the installed script:
retinanet-train pascal <path to VOCdevkit/VOC2007>
```

For training on [MS COCO](http://cocodataset.org/#home), run:
```
# Running directly from the repository:
keras_retinanet/bin/train.py coco <path to MS COCO>

# Using the installed script:
retinanet-train coco <path to MS COCO>
```

For training on a custom dataset, a CSV file can be used as a way to pass the data.
See below for more details on the format of these CSV files.
To train using your CSV, run:
```
# Running directly from the repository:
keras_retinanet/bin/train.py csv <path to csv file containing annotations> <path to csv file containing classes>

# Using the installed script:
retinanet-train csv <path to csv file containing annotations> <path to csv file containing classes>
```

In general, the steps to train on your own datasets are:
1) Create a model by calling for instance `keras_retinanet.models.ResNet50RetinaNet` and compile it.
   Empirically, the following compile arguments have been found to work well:
```
model.compile(
    loss={
        'regression'    : keras_retinanet.losses.regression_loss,
        'classification': keras_retinanet.losses.focal_loss()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)
```
2) Create generators for training and testing data (an example is show in [`keras_retinanet.preprocessing.PascalVocGenerator`](https://github.com/Xpitfire/keras-retinanet/blob/master/keras_retinanet/preprocessing/pascal_voc.py)).
3) Use `model.fit_generator` to start training.

## Testing
An example of testing the network can be seen in [this Notebook](https://github.com/Xpitfire/keras-retinanet/blob/master/ResNet50RetinaNet%20-%20COCO%202017.ipynb).
In general, output can be retrieved from the network as follows:
```
_, _, detections = model.predict_on_batch(inputs)
```

Where `detections` are the resulting detections, shaped `(None, None, 4 + num_classes)` (for `(x1, y1, x2, y2, cls1, cls2, ...)`).

Loading models can be done in the following manner:
```
from keras_retinanet.models.resnet import custom_objects
model = keras.models.load_model('/path/to/model.h5', custom_objects=custom_objects)
```

Execution time on NVIDIA Pascal Titan X is roughly 55msec for an image of shape `1000x600x3`.

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

A full example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
```

This defines a dataset with 3 images.
`img_001.jpg` contains a cow.
`img_002.jpg` contains a cat and a bird.
`img_003.jpg` contains no interesting objects/animals.


### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```

## Results

### MS COCO
The MS COCO model can be downloaded [here](https://1drv.ms/u/s!Ai9oaxqJ6sUumNJLUn8AGxnSMnaLvA). Results using the `cocoapi` are shown below (note: according to the paper, this configuration should achieve a mAP of 0.343).

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.513
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.342
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.149
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.354
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.437
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.263
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.510
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.623
```

## Status
Example output images using `keras-retinanet` are shown below.

<p align="center">
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco1.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco2.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco3.png" alt="Example result of RetinaNet on MS COCO"/>
</p>

### Notes
* This repository requires Keras 2.1.2.
* This repository is tested using OpenCV 3.3.
* Warnings such as `UserWarning: Output "non_maximum_suppression_1" missing from loss dictionary.` can safely be ignored. These warnings indicate no loss is connected to these outputs, but they are intended to be outputs of the network for the user (ie. resulting network detections) and not loss outputs.

Contributions to this project are welcome.

