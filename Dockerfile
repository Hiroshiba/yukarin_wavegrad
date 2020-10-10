FROM hiroshiba/hiho-deep-docker-base:pytorch1.5.0-cuda9.0

SHELL ["/bin/bash", "-c"]
WORKDIR /app

# install requirements
COPY requirements.txt /app/
RUN pip install -r <(cat requirements.txt | grep -v torch)
