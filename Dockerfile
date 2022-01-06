FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
LABEL maintainer="Naoki Katsura(https://github.com/katsura-jp) <nok.addayo@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    zip unzip \
    htop tree \
    wget curl \
    vim ssh tmux \
    libopencv-dev \
    build-essential libssl-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libffi-dev \
    dpkg-dev \
    python3 python3-pip python3-dev \
    default-jre \
    ffmpeg

RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*


ARG UID
RUN useradd docker -l -u $UID -G sudo -s /bin/bash -m
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo 'docker ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER docker

ENV PYENV_ROOT /home/docker/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

ENV PYTHON_VERSION 3.8.6
RUN pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}


RUN pip install -U pip setuptools 
RUN pip install torch==1.7.1 torchvision==0.8.2
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

WORKDIR /opt/ml

ENTRYPOINT [ "/bin/bash" ]