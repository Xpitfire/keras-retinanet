#FROM nvidia/cuda:9.1-devel-ubuntu16.04
#LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

#ENV CUDNN_VERSION 7.0.5.15
#LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

#RUN apt-get update && apt-get install -y --no-install-recommends \
#            libcudnn7=$CUDNN_VERSION-1+cuda9.1 \
#            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.1 && \
#    rm -rf /var/lib/apt/lists/*
FROM ubuntu:latest
#FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y python3 python3-setuptools python3-pip python3-dev python-opencv \ 
                                         build-essential libssl-dev libffi-dev git
RUN pip3 install --upgrade pip

#RUN apt-get update && apt-get install -y python-opencv build-essential libssl-dev libffi-dev git
#RUN pip install --upgrade pip

ADD . /code
WORKDIR /code

RUN pip3 install -r requirements.txt
RUN pip3 install --user --upgrade git+https://github.com/broadinstitute/keras-resnet
RUN python3 setup.py install --user
#RUN cd keras_retinanet/preprocessing
#RUN python3 setup.py build_ext --inplace
#RUN python3 setup.py install

#RUN pip install -r requirements.txt
#RUN pip install --user --upgrade git+https://github.com/broadinstitute/keras-resnet
#RUN python setup.py install --user
#RUN cd keras_retinanet/preprocessing
#RUN python setup.py build_ext --inplace
#RUN python setup.py install

COPY ./wrapper_script.sh /
CMD ["/wrapper_script.sh"]