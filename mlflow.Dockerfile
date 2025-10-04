# mlflow.Dockerfile
FROM python:3.12-slim

UN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Bangkok

RUN apt update && apt upgrade -y \
    && apt install -y tzdata locales curl \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en

WORKDIR /mlflow

RUN pip3 install --upgrade pip
RUN pip3 install mlflow

# Install MLflow only
RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--gunicorn-opts", "--workers 2 --timeout 300"]

