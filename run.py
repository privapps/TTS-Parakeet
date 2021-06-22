import numpy as np
import paddle
from parakeet.utils import display
from parakeet.utils import layer_tools

import sys
sys.path.append("../..")
import examples
from parakeet.models.tacotron2 import Tacotron2
from parakeet.frontend import EnglishCharacter
from examples.tacotron2 import config as tacotron2_config

# text to bin model
synthesizer_config = tacotron2_config.get_cfg_defaults()
synthesizer_config.merge_from_file("../../../tacotron2_ljspeech_ckpt_0.3/config.yaml")
frontend = EnglishCharacter()
model = Tacotron2.from_pretrained(
    synthesizer_config, "../../../tacotron2_ljspeech_ckpt_0.3/step-179000")
model.eval()


from parakeet.models.waveflow import ConditionalWaveFlow
from examples.waveflow import config as waveflow_config

# bin to wav model
vocoder_config = waveflow_config.get_cfg_defaults()
vocoder = ConditionalWaveFlow.from_pretrained(
    vocoder_config,
    "../../../waveflow_ljspeech_ckpt_0.3/step-2000000")
layer_tools.recursively_remove_weight_norm(vocoder)
vocoder.eval()

def one_by_one(line, arr):
    text = line.strip()
    if(len(text)<=1):
        return
    sentence = paddle.to_tensor(frontend(line)).unsqueeze(0)

    with paddle.no_grad():
        outputs = model.infer(sentence)
    mel_output = outputs["mel_outputs_postnet"][0].numpy().T
    alignment = outputs["alignments"][0].numpy().T

    #now bin to audio
    audio = vocoder.infer(paddle.transpose(outputs["mel_outputs_postnet"], [0, 2, 1]))
    arr.append(audio[0].numpy())

samplerate = 22050
    
def generate_blank(seconds : float):    
    samples = int(samplerate * seconds) # aka samples per second
    return np.resize( 0, ( samples, ) )

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

with open('/workspace/__input__.txt', 'rt') as f:
    lines = f.readlines()

npwav = []
out_part=[]
out_prefix='/workspace/p_'
from scipy.io.wavfile import write as wvwrite
for paragraph in lines:
    for line in tokenizer.tokenize(paragraph):
        try:
            if len(line) < 1:
                continue
            if len(nltk.word_tokenize(line)) > 25:
                for l in line.split(','): # sentence too long, cut manually
                    one_by_one(l, npwav)
                    npwav.append(generate_blank(0.3))
            else:
                one_by_one(line, npwav)
                npwav.append(generate_blank(0.5))
            # wav to part file
            file_name = out_prefix + str(len(out_part)) + '.wav'
            wvwrite(file_name, samplerate, np.concatenate(npwav))
            out_part.append(file_name)
            npwav = []
        except Exception as inst:
            print("Unexpected error:", sys.exc_info()[0])
    npwav.append(generate_blank(0.8))
if len(npwav) > 1:
    file_name = out_prefix + str(len(out_part)) + '.wav'
    out_part.append(file_name)
    wvwrite(file_name, samplerate, np.concatenate(npwav))

npwav = []
file_name='/workspace/__out__.wav'

from scipy.io import wavfile
for i in out_part:
    rate, data = wavfile.read(i)
    npwav.append(data)
wvwrite(file_name, samplerate, np.concatenate(npwav))