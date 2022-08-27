import numpy as np
import os
import audio


def build_from_path(in_dir, out_dir):
    index = 1
    texts = []

    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f.readlines():
            if index % 100 == 0:
                print("{:d} Done".format(index))
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            idx = parts[0]
            #print("Part",parts)
            texts.append("|".join([idx,_process_utterance(out_dir, index, wav_path, text)]))

            index += 1

    return texts


def _process_utterance(out_dir, index, wav_path, text):
    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.tools.get_mel(wav_path).numpy().astype(np.float32)

    # Write the spectrograms to disk:
    mel_filename = 'ljspeech-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.T, allow_pickle=False)

    return text

import string

def get_idx_dict(in_dir):
    index = 0
    text_dict = {}

    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f.readlines():
            # if index % 100 == 0:
            #     print("{:d} Done".format(index))
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            idx = parts[0]
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = text.lower()
            #print("Part",parts)
            text_dict.update({index:{"idx":idx,"text":text,"wav_path":wav_path}})

            index += 1

    return text_dict
