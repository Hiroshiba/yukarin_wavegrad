FROM hiroshiba/hiho-deep-docker-base:pytorch1.5.0-cuda9.0
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y swig libsndfile1-dev libasound2-dev && \
    apt-get clean

WORKDIR /app

# install requirements
COPY requirements.txt /app/
RUN pip install -r <(cat requirements.txt | grep -v 'torch==1.5.0')
