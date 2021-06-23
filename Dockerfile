FROM privapps/tts-parakeet:0.3_base

RUN python -c "import nltk;nltk.download('punkt');nltk.data.load('tokenizers/punkt/english.pickle')"

COPY run.sh /
COPY run.py /opt/Parakeet/examples/tacotron2
RUN mkdir -p $WORKDIR ; chmod a+rwx $WORKDIR run.sh ; apt install -y lame

ENTRYPOINT ["/run.sh"]
