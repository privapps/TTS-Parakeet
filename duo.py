import numpy as np
import paddle
from parakeet.utils import display
from parakeet.utils import layer_tools

import sys, os, re
sys.path.append("..")
import examples
from examples.tacotron2 import config as tacotron2_config
from examples.transformer_tts import config as transformer_config
from parakeet.models.tacotron2 import Tacotron2
from parakeet.models.transformer_tts import TransformerTTS
from parakeet.frontend import EnglishCharacter
from parakeet.frontend import English
from examples.tacotron2 import config as tacotron2_config
from parakeet.models.waveflow import ConditionalWaveFlow
from examples.waveflow import config as waveflow_config



# text to bin model
def init_tacotron2_model():
    synthesizer_tacotron2_config = tacotron2_config.get_cfg_defaults()
    synthesizer_tacotron2_config.merge_from_file("../../tacotron2_ljspeech_ckpt_0.3/config.yaml")
    tacotron2_model = Tacotron2.from_pretrained(
        synthesizer_tacotron2_config, "../../tacotron2_ljspeech_ckpt_0.3/step-179000")
    tacotron2_model.eval()
    return tacotron2_model

#transformer to bin model
def init_transformer_model():
    synthesizer_transformer_config = transformer_config.get_cfg_defaults()
    synthesizer_transformer_config.merge_from_file("../../transformer_tts_ljspeech_ckpt_0.3/config.yaml")
    transformer_model = TransformerTTS.from_pretrained(
        transformer_frontend, synthesizer_transformer_config, "../../transformer_tts_ljspeech_ckpt_0.3/step-310000")
    transformer_model.eval()
    return transformer_model



# bin to wav model
def init_vocoder_model():
    vocoder_config = waveflow_config.get_cfg_defaults()
    vocoder = ConditionalWaveFlow.from_pretrained(
        vocoder_config,
        "../../waveflow_ljspeech_ckpt_0.3/step-2000000")
    layer_tools.recursively_remove_weight_norm(vocoder)
    vocoder.eval()
    return vocoder

import nltk.data
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def one_by_one(line, arr):
    text = line.strip()
    if(len(text)<=1):
        return
    long_words = [wrd for wrd in nltk.word_tokenize(text) if len(wrd) > 3]
    print('^^long_words^^', long_words, text)
    if len(long_words) > 3 and not re.search(r'([A-Z]\.)+', text):
        print('debug',text, nltk.word_tokenize(text))
        audio_data = _tacotron2_one(text)
    else:
        audio_data = _transformer_one(text)
    arr.append(audio_data)

def do_long_sentence(line, npwav):
    # smart to split it into long sentences
    commas_split_positions = []
    last_split_words_count = 0
    just_past_comma_words_count=0
    just_past_comma_start=0
    for i,c in enumerate(line):
        if ',' == c:
            current_words_count = len(nltk.word_tokenize(line[0:i]))
            if current_words_count - last_split_words_count > 25:
                commas_split_positions.append(just_past_comma_start)
                last_split_words_count = just_past_comma_words_count
                last_words_start = just_past_comma_start
            just_past_comma_start = i
            just_past_comma_words_count = current_words_count
    if len(nltk.word_tokenize(line)) - last_split_words_count > 25:
        commas_split_positions.append(just_past_comma_start)
    o_words = []
    w_begin=0
    for pos in commas_split_positions: 
        o_words.append(line[w_begin:pos])
        w_begin=pos + 1
    if(w_begin<len(line)):
        o_words.append(line[w_begin:])

    for word in o_words:
        one_by_one(word, npwav)
        npwav.append(generate_blank(0.2))

def _tacotron2_one(line : str):
    print('$$',line,'$$')
    sentence = paddle.to_tensor(tacotron2_frontend(line.strip())).unsqueeze(0)
    with paddle.no_grad():
        outputs = tacotron2_model.infer(sentence)
    mel_output = outputs["mel_outputs_postnet"][0].numpy().T
    alignment = outputs["alignments"][0].numpy().T

    #now bin to audio
    audio = vocoder.infer(paddle.transpose(outputs["mel_outputs_postnet"], [0, 2, 1]))
    return audio[0].numpy()


def _transformer_one(line : str):
    print('@@',line,'@@')
    text = paddle.to_tensor(transformer_frontend(line.strip()))
    text = paddle.unsqueeze(text, 0)  # (1, T)

    with paddle.no_grad():
        outputs = transformer_model.infer(text, verbose=True)
    audio = vocoder.infer(paddle.transpose(outputs["mel_output"], [0, 2, 1]))
    wav = audio[0].numpy().T #(C, T)
    # in case spiked noise
    if np.max(wav) >= 1:
        pc=np.percentile(wav, 99.99)
        for po in np.argwhere((wav>pc) | (wav<-pc)):
            index=po[0]
            wav[max(0,index-50):min(len(po),index+50)]=0
    return wav


SAMPLE_RATE = 22050
OUTPUT_PREFIX='/workspace/p_'

def generate_blank(seconds : float):
    samples = int(SAMPLE_RATE * seconds) # aka samples per second
    return np.resize( 0, ( samples, ))

from scipy.io.wavfile import write as wvwrite
def write_multi_wav_file(lines):
    npwav = []
    out_part=[]
    for paragraph in lines:
        for line in tokenizer.tokenize(paragraph):
            print('paragraph <<<',line, '>>>')
            try:
                if len(line) < 1:
                    continue
                words = len(nltk.word_tokenize(line))
                if words < 3: # sentence too short, has to use transformer
                    npwav.append(_transformer_one(line))
                elif words > 25:
                    do_long_sentence(line, npwav)
                else:
                    one_by_one(line, npwav)

                npwav.append(generate_blank(0.5))
                # wav to part file
                file_name = OUTPUT_PREFIX + str(len(out_part)) + '.wav'
                wvwrite(file_name, SAMPLE_RATE, np.concatenate(npwav))
                out_part.append(file_name)
                npwav = []
            except Exception as inst:
                print("Unexpected error:", sys.exc_info())
        npwav.append(generate_blank(0.8))

    if len(npwav) > 1:
        file_name = OUTPUT_PREFIX + str(len(out_part)) + '.wav'
        out_part.append(file_name)
        wvwrite(file_name, SAMPLE_RATE, np.concatenate(npwav))
    return out_part

### main
tacotron2_frontend = EnglishCharacter()
tacotron2_model = init_tacotron2_model()
transformer_frontend = English()
transformer_model = init_transformer_model()
vocoder = init_vocoder_model()

lines = os.environ['INPUT_TEXT'].split('\n')
out_part=write_multi_wav_file(lines)

npwav = []
file_name='/workspace/__out__.wav'

from scipy.io import wavfile
for i in out_part:
    rate, data = wavfile.read(i)
    npwav.append(data)


if len(npwav) > 1 :
    wvwrite(file_name, SAMPLE_RATE, np.concatenate(npwav))
else:
    wvwrite(file_name, SAMPLE_RATE, npwav[0])
