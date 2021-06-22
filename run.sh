#!/bin/bash
cd /opt/Parakeet/examples/tacotron2/
python run.py
OUT_FILE=$WORKDIR/__out__
lame ${OUT_FILE}.wav ${OUT_FILE}.mp3
rm -f $WORKDIR/p_*.wav
