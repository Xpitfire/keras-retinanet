FROM nvidia/cuda:8.0-cudnn6-devel

# -------------------------------------------
# General packages
# -------------------------------------------
RUN apt-get update && apt-get install -y python3 python3-setuptools python3-pip python3-dev python-opencv \ 
                                         build-essential libssl-dev libffi-dev git wget module-init-tools libcupti-dev
RUN pip3 install --upgrade pip

# -------------------------------------------
# Workspace reference
# -------------------------------------------
ADD . /code
WORKDIR /code

# -------------------------------------------
# Python tools setup
# -------------------------------------------
RUN pip3 install -r requirements.txt
RUN pip3 install --user --upgrade git+https://github.com/broadinstitute/keras-resnet
RUN python3 setup.py install --user
# setting notebook options
RUN mkdir ~/.jupyter
RUN touch ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = u''" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = u''" >> ~/.jupyter/jupyter_notebook_config.py

# -------------------------------------------
# Startup script
# -------------------------------------------
COPY ./wrapper_script.sh /
CMD ["/wrapper_script.sh"]
