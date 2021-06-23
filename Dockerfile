FROM privapps/tts-parakeet:latest


COPY action.sh /
COPY prepare.py /

RUN chmod +x /action.sh

ENTRYPOINT ["/action.sh"]
