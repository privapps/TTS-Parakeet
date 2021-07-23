#!/bin/bash

_PWD=`pwd`
cd /opt/Parakeet/examples/
python duo.py
OUT_FILE=$WORKDIR/__out__
lame -V 0 ${OUT_FILE}.wav ${_PWD}/$INPUT_CONTENT
rm -f $WORKDIR/*.wav
mv $WORKDIR/*.npy ${_PWD}
