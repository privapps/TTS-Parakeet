FROM privapps/tts-parakeet:duo


COPY action.sh /
COPY duo.py /opt/Parakeet/examples/

RUN chmod +x /action.sh

ENTRYPOINT ["/action.sh"]
