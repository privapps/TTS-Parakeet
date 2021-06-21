FROM ubuntu:20.04

RUN apt-get update -y --fix-missing && apt-get install -y libsndfile1 libsndfile1 unzip wget git python3 pip && pip install paddle-parakeet paddlepaddle scipy nltk
RUN cd /usr/bin ; ln -s python3 python


ENV WORKDIR /workspace

RUN cd /opt/ ;\
    wget -q https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_ckpt_0.3.zip >/dev/null && unzip tacotron2_ljspeech_ckpt_0.3.zip && rm -f tacotron2_ljspeech_ckpt_0.3.zip & \
    wget -q https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_ljspeech_ckpt_0.3.zip >/dev/null &&  unzip waveflow_ljspeech_ckpt_0.3.zip && rm -f waveflow_ljspeech_ckpt_0.3.zip & \
    git clone --depth 1 https://github.com/iclementine/Parakeet & \
    wait


RUN python -c "import nltk.data; nltk.download('punkt')" 

ENTRYPOINT ["bash"]
