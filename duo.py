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

from scipy.io.wavfile import write as wvwrite
from scipy.io import wavfile

SAMPLE_RATE = 22050
OUTPUT_PREFIX = '/workspace/p_'


# text to bin model
def init_tacotron2_model():
    synthesizer_tacotron2_config = tacotron2_config.get_cfg_defaults()
    synthesizer_tacotron2_config.merge_from_file("../../tacotron2_ljspeech_ckpt_0.3/config.yaml")
    tacotron2_model = Tacotron2.from_pretrained(
        synthesizer_tacotron2_config, "../../tacotron2_ljspeech_ckpt_0.3/step-179000")
    tacotron2_model.eval()
    return tacotron2_model


# transformer to bin model
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
    if len(text) <= 1 or not re.search(r'[a-zA-Z0-9]+', text):
        return
    long_words = [wrd for wrd in nltk.word_tokenize(text) if len(wrd) > 3]
    if len(long_words) > 3 and _should_tacotron2(text):
        audio_data = _tacotron2_one(text)
    else:
        audio_data = _transformer_one(text)
    arr.append(audio_data)


def _should_tacotron2(text: str):
    return not re.search(r'([A-Z]\.)+', text) \
           and not re.search(r'Chin(a|ese)', text) \
           and not re.search(r'(?i)focus(es|ing|ed)', text) \
           and not re.search(r'(?i)(rational|anger|imagine|island|chaos)', text) \
           and text.find('é') < 0 \
           and not re.search(r'Diane|Christ|RAHN|Jesus', text)


def do_long_sentence(line, npwav):
    # smart to split it into long sentences
    commas_split_positions = []
    last_split_words_count = 0
    just_past_comma_words_count = 0
    just_past_comma_start = 0
    for i, c in enumerate(line):
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
    w_begin = 0
    for pos in commas_split_positions:
        o_words.append(line[w_begin:pos])
        w_begin = pos + 1
    if (w_begin < len(line)):
        o_words.append(line[w_begin:])

    for word in o_words:
        one_by_one(word, npwav)
        npwav.append(generate_blank(0.15))


def _tacotron2_one(line: str):
    print('$$', line, '$$')
    sentence = paddle.to_tensor(tacotron2_frontend(line.strip())).unsqueeze(0)
    algm, mel_output = _tacotron2_satisfy(sentence)
    _save_tensor(line, mel_output, 'a_', algm)
    # now bin to audio
    audio = vocoder.infer(paddle.transpose(mel_output, [0, 2, 1]))
    return audio[0].numpy()


def _tacotron2_gen(sentence: str):
    with paddle.no_grad():
        outputs = tacotron2_model.infer(sentence, max_decoder_steps=1500)
        return (_get_alignmnet(outputs), outputs["mel_outputs_postnet"])


def _tacotron2_satisfy(sentence: str, threashhold=0.0015, max_num=10):
    keep = []
    for i in range(0, max_num):
        algm, tensor = _tacotron2_gen(sentence)
        if algm < threashhold:
            return (algm, tensor)
        keep.append((algm, tensor))
    min = 0
    for idx, element in enumerate(keep):
        if element[0] < keep[min][0]:
            min = idx
    return keep[min]


def _get_alignmnet(outputs):
    alignments = outputs["alignments"][0].numpy()
    x, y = np.shape(alignments)
    return alignments[x - 1][y - 1]


def _transformer_one(line: str):
    print('@@', line, '@@')
    text = paddle.to_tensor(transformer_frontend(line.strip()))
    text = paddle.unsqueeze(text, 0)  # (1, T)

    with paddle.no_grad():
        outputs = transformer_model.infer(text, max_length=1500)
    mel_output = outputs["mel_output"]
    _save_tensor(line, mel_output, 'r_', '')

    audio = vocoder.infer(paddle.transpose(mel_output, [0, 2, 1]))
    wav = audio[0].numpy().T  # (C, T)

    # in case spiked noise at the end
    check_po = len(wav) - 600
    pc = np.percentile(wav[:check_po], 99)
    sub_wav = wav[check_po:]
    po = np.argwhere((sub_wav > pc) | (sub_wav < -pc))

    if len(po) > 0 and len(po[0]) > 0:
        index = po[0][0] + check_po
        wav[max(check_po, index - 300):min(len(wav), index + 300)] = 0
    return wav


def generate_blank(seconds: float):
    samples = int(SAMPLE_RATE * seconds)  # aka samples per second
    return np.resize(0, (samples,))


def write_multi_wav_file(lines):
    npwav = []
    out_part = []
    for paragraph in lines:
        for line in tokenizer.tokenize(paragraph):
            print('paragraph <<<', line, '>>>')
            try:
                if len(line) < 1:
                    npwav.append(generate_blank(.9))
                    continue
                words = len(nltk.word_tokenize(line))
                if words < 3:  # sentence too short, has to use transformer
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
        # npwav.append(generate_blank(0.7))

    if len(npwav) > 1:
        file_name = OUTPUT_PREFIX + str(len(out_part)) + '.wav'
        out_part.append(file_name)
        wvwrite(file_name, SAMPLE_RATE, np.concatenate(npwav))
    return out_part


def _save_tensor(line: str, tensor, prefix: str, algm):
    f_n = OUTPUT_PREFIX + prefix + re.sub("[^a-z0-9]+", "_", line.lower()) + '_' + str(algm) + '.npy'
    np.save(f_n, tensor.numpy())


### main
tacotron2_frontend = EnglishCharacter()
tacotron2_model = init_tacotron2_model()
transformer_frontend = English()
transformer_model = init_transformer_model()
vocoder = init_vocoder_model()

lines = os.environ['INPUT_TEXT'].split('\n')
out_part = write_multi_wav_file(lines)

npwav = []
file_name = '/workspace/__out__.wav'

for i in out_part:
    rate, data = wavfile.read(i)
    npwav.append(generate_blank(0.7))
    npwav.append(data)

if len(npwav) > 1:
    wvwrite(file_name, SAMPLE_RATE, np.concatenate(npwav))
else:
    wvwrite(file_name, SAMPLE_RATE, npwav[0])
