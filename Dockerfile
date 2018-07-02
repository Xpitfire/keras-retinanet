FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04

# -------------------------------------------
# Set Environment
# -------------------------------------------
ENV PATH /opt/conda/bin:$PATH

# -------------------------------------------
# General packages
# -------------------------------------------
RUN apt-get update && apt-get install -y build-essential libssl-dev libffi-dev git wget module-init-tools libcupti-dev

# -------------------------------------------
# Python tools setup
# -------------------------------------------
# install anaconda for
RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
RUN conda update conda

# chose between CPU or GPU version:
RUN conda install -y -c pytorch faiss-gpu
#RUN conda install -y -c pytorch faiss-cpu

# update pip package manager
RUN pip install --upgrade pip

# change to workspace
ADD . /code
WORKDIR /code

# install pip requirements
RUN pip install -r requirements.txt

# install additional python packages
RUN apt-get install -y python-opencv python-setuptools python-dev

# -------------------------------------------
# custom keras tools
# -------------------------------------------
RUN pip install . --user

# setting notebook options for development mode -> uncomment for production mode
RUN mkdir ~/.jupyter
RUN touch ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = u''" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = u''" >> ~/.jupyter/jupyter_notebook_config.py

# -------------------------------------------
# Startup script
# -------------------------------------------
EXPOSE 5000
EXPOSE 9090

COPY ./wrapper_script.sh /
CMD ["/wrapper_script.sh"]
