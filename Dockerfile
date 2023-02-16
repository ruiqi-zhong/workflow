# Build using the standard pytorch base image

FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04 as dev_base

WORKDIR .
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y
RUN apt install python3-pip -y
RUN apt-get install -qq libopenmpi-dev -y
RUN apt-get install magic-wormhole -y
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install nltk
RUN pip install -U scikit-learn
RUN pip install scipy
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader averaged_perceptron_tagger

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# RUN chmod +x /usr/bin/tini
# ENTRYPOINT ["/usr/bin/tini", "--"]
