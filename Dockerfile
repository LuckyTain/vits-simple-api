FROM artrajz/pytorch:1.13.1-cu117-py3.10.11-ubuntu22.04

RUN mkdir -p /app
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -yq build-essential espeak-ng cmake wget ca-certificates tzdata&& \
    update-ca-certificates && \
    apt-get clean && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/* 


# Install jemalloc
RUN wget https://github.com/jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2 && \
    tar -xvf jemalloc-5.3.0.tar.bz2 && \
    cd jemalloc-5.3.0 && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf jemalloc-5.3.0* && \
    ldconfig

ENV LD_PRELOAD=/usr/local/lib/libjemalloc.so

COPY requirements.txt /app/
RUN pip install gunicorn --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir&& \
    rm -rf /root/.cache/pip/*

RUN mkdir -p Model/acg && \
    wget https://huggingface.co/spaces/TLME/Bert-VITS-Umamusume-Genshin-HonkaiSR/resolve/main/logs/UGH/G_100000.pth -O Model/acg/G_100000.pth && \
    wget https://huggingface.co/spaces/TLME/Bert-VITS-Umamusume-Genshin-HonkaiSR/resolve/main/configs/config.json -O Model/acg/config.json && \
    wget https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin -O bert_vits2/bert/chinese-roberta-wwm-ext-large/pytorch_model.bin && \
    wget https://huggingface.co/cl-tohoku/bert-base-japanese-v3/resolve/main/pytorch_model.bin -O bert_vits2/bert/bert-base-japanese-v3/pytorch_model.bin

COPY . /app

EXPOSE 23456
EXPOSE 8000

CMD python3 -u banana_app.py
#CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]