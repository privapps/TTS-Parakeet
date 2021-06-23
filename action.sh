#!/bin/bash

python /prepare.py
bash -c /run.sh

if [ "${INPUT_FORMART}" == "wav" ]; then
	mv $WORKDIR/__out__.wav $INPUT_CONTENT
else
	mv $WORKDIR/__out__.mp3 $INPUT_CONTENT
fi