FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3 python3-setuptools python3-pip python3-dev python-opencv \ 
                                         build-essential libssl-dev libffi-dev git
RUN pip3 install --upgrade pip

ADD . /code
WORKDIR /code

RUN pip3 install -r requirements.txt
RUN pip3 install --user --upgrade git+https://github.com/broadinstitute/keras-resnet
RUN python3 setup.py install --user

COPY ./wrapper_script.sh /
CMD ["/wrapper_script.sh"]