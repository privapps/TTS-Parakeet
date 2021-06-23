FROM privapps/tts-parakeet:0.3_base

RUN python -c "import nltk;nltk.download('punkt');tokenizer=nltk.data.load('tokenizers/punkt/english.pickle');tokenizer.tokenize('b')" & \
    mkdir -p $WORKDIR ; chmod a+rwx $WORKDIR; \
    cd /opt/ && \
    wget -q https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_ckpt_0.3.zip >/dev/null && unzip transformer_tts_ljspeech_ckpt_0.3.zip && rm -f transformer_tts_ljspeech_ckpt_0.3.zip & \
    apt install -y lame & \
    wait


ENTRYPOINT ["date"]
