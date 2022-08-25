# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import torch
import torch.utils.data

import tacotron2_common.layers as layers
from tacotron2_common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu
from text import text_to_sequence
import librosa


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, dataset_path, audiopaths_and_text, args, vocab = None, tokenizer = None, print_raw_text=False):
        self.audiopaths_and_text = load_filepaths_and_text(dataset_path, audiopaths_and_text)
        self.text_cleaners = args.text_cleaners
        self.max_wav_value = args.max_wav_value
        self.sampling_rate = args.sampling_rate
        self.load_mel_from_disk = args.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.sampling_rate, args.mel_fmin,
            args.mel_fmax)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.print_raw_text = print_raw_text

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        #print("current path",audiopath_and_text)
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        len_text = len(text)
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel, len_text)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.load(filename)
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm
        # if self.vocab and self.tokenizer:
        #     if self.print_raw_text:
        #         print("Raw text", text)
        #     text_norm = torch.tensor(self.vocab(self.tokenizer(text)), dtype=torch.long)
        #     #print("shape", text_norm.size())
        #     #print("Norm text",text_norm)
        #     return text_norm
        # else:
        #     return text

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        #print("IDS",input_lengths,ids_sorted_decreasing)
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            #print("IDS i mel size",i, mel.size(1))
            gate_padded[i, mel.size(1)-1:] = 1
            #print("GGGGGGGate", gate_padded)
            output_lengths[i] = mel.size(1)

        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)
        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, len_x

def batch_to_gpu(batch):
    text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, len_x = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
    y = (mel_padded, gate_padded)
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)




def get_collate_function(model_name, n_frames_per_step=1):
    if model_name == 'Tacotron2':
        collate_fn = TextMelCollate(n_frames_per_step)
    elif model_name == 'WaveGlow':
        collate_fn = torch.utils.data.dataloader.default_collate
    else:
        raise NotImplementedError(
            "unknown collate function requested: {}".format(model_name))

    return collate_fn


def get_data_loader(model_name, dataset_path, audiopaths_and_text, args):
    if model_name == 'Tacotron2':
        data_loader = TextMelLoader(dataset_path, audiopaths_and_text, args)
    else:
        raise NotImplementedError(
            "unknown data loader requested: {}".format(model_name))

    return data_loader

import numpy as np
import math
import time
import os
from torch.utils.data import Dataset, DataLoader
from utils import process_text, pad_1D, pad_2D
import hparams
from tqdm import tqdm
from utils import pad_1D_tensor, pad_2D_tensor
from data.ljspeech import get_idx_dict
from glottal_flow import get_rosenberg_waveform


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]
    stft_volume_velocity_list = [torch.Tensor(batch[ind]["stft_volume_velocity"].T) for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    #print("Mel",mel_targets.size())
    mel_targets = pad_2D_tensor(mel_targets)
    stft_volume_velocity_targets = pad_2D_tensor(stft_volume_velocity_list)
    #print("STFT VV",stft_volume_velocity_targets.size())
    if stft_volume_velocity_targets.size(1) > mel_targets.size(1):
        stft_volume_velocity_targets = stft_volume_velocity_targets[:,:mel_targets.size(1),:]
    if stft_volume_velocity_targets.size(1) < mel_targets.size(1):
        stft_padding = torch.zeros(stft_volume_velocity_targets.size(0),
                                   mel_targets.size(1) - stft_volume_velocity_targets.size(1),
                                   stft_volume_velocity_targets.size(2))
        stft_volume_velocity_targets = torch.cat((stft_volume_velocity_targets,stft_padding),dim=1)
    #print("STFT VV padded", stft_volume_velocity_targets.size())

    out = {"text": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len,
           "stft_volume_velocity":stft_volume_velocity_targets}

    return out

def collate_fn_tensor(batch):
    len_arr = np.array([d["text"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // hparams.batch_expand_size

    cut_list = list()
    for i in range(hparams.batch_expand_size):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(hparams.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output



def get_phoneme_type(path):
    phoneme_type_dict = {}
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            l = line.split()
            phoneme = l[0]
            phoneme_type = l[1]
            phoneme_type_dict.update({phoneme:phoneme_type})
    #print("type dict",phoneme_type_dict)
    return phoneme_type_dict

def get_phoneme(path,phoneme_type_dict):
    vocab_phoneme_dict = {}
    phonme_total = set()
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            l = line.split()
            vocab = l[0].lower()
            phonemes = l[1:]
            phoneme_info_list = []
            previous_type = "common"
            for phoneme in phonemes:
                phonme_total.add(phoneme)
                phoneme_type = phoneme_type_dict[phoneme] if phoneme in phoneme_type_dict else previous_type
                previous_type = phoneme_type
                #print("P_type",phoneme_type)
                phoneme_info = {"phoneme":phoneme,"phoneme_type": phoneme_type,"config":hparams.phoneme_config[phoneme_type]}
                phoneme_info_list.append(phoneme_info)
            if vocab in vocab_phoneme_dict:
                vocab_phoneme_dict[vocab]['phoneme'].append(phoneme_info_list)
            else:
                vocab_phoneme_dict[vocab] = {"phoneme":[phoneme_info_list]}
    #print("Vocab dict",vocab_phoneme_dict.keys())
    return vocab_phoneme_dict, phonme_total

def get_sli_info(path):
    sli_dict = {}
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            l = line.split()
            idx = l[0]
            start = float(l[2])/100
            duration = float(l[3])/100
            vocab = l[4]
            if idx in sli_dict:
                sli_dict[idx]["vocabs"].append(vocab)
                sli_dict[idx]["starts"].append(start)
                sli_dict[idx]["durations"].append(duration)
            else:
                sli_dict[idx] = {"vocabs":[vocab],"starts":[start],"durations":[duration]}
    return sli_dict



def get_prior_phoneme_sepctrogram_info(top_n=2500):
    """
    return dict: key is the index of the train set, value is the spectrogram
    """
    meta_path = os.path.join("data", "LJSpeech-1.1")
    idx_dict = get_idx_dict(meta_path)

    text_list = list(map(lambda x:x, idx_dict.items()))
    text_list.sort(key=lambda x:x[0])
    top_n_idx = set(map(lambda x:x[1]["idx"],text_list[:top_n]))


    phoneme_type_path = hparams.cmu_phoneme_type_path
    phoneme_type_dict = get_phoneme_type(phoneme_type_path)

    phoneme_path = hparams.cmu_phoneme_path
    vocab_phoneme_info,phoneme_total_set = get_phoneme(phoneme_path,phoneme_type_dict)

    sli_path = hparams.cmu_sli_path
    sli_info = get_sli_info(sli_path)



    waveform_dict = {}
    stft_dict = {}
    not_in_file = set()
    finish_count = 0
    for k,v in sli_info.items():
        idx = k
        if idx not in top_n_idx:
            continue

        ronsen_path = os.path.join("data","rosen_waveform", "%s_waveform.npy"%idx)


        try:
            ronsen_stft_abs = np.load(ronsen_path)
            stft_dict[idx] = ronsen_stft_abs
            continue
        except:
            print("Disk file not available, using calculated results")




        vocabs = v["vocabs"]
        durations = v["durations"]
        waveforms = []
        for index,vocab in enumerate(vocabs):
            duration = durations[index]
            if "_sil" in vocab:
                waveform = get_rosenberg_waveform(duration=duration,is_voiced=False,sr = 22500,t_0=1/125,A=4/1000,t_p=0.8*1/125 * 2/3,t_n=0.8*1/125 * 1/3,O_q=0.8)
                #print("Wave pice",vocab,len(waveform))
                waveforms += waveform
            else:
                if vocab not in vocab_phoneme_info:
                    #print("Not in file ", vocab, "current loop",finish_count)
                    not_in_file.add(vocab)
                    waveform = get_rosenberg_waveform(duration=phoneme_duration)
                    #print("Wave pice", vocab,len(waveform))
                    waveforms += waveform
                else:
                    vocab_ = vocab_phoneme_info[vocab]
                    phonemes = vocab_["phoneme"][0]  # if multi phoneme, chose the first.
                    phoneme_duration = duration/len(phonemes)
                    for phoneme in phonemes:
                        phoneme_config = phoneme['config']
                        waveform = get_rosenberg_waveform(duration = phoneme_duration,**phoneme_config)
                        #print("Wave pice",vocab, phoneme,phoneme_duration,duration,len(waveform))
                        waveforms += waveform
        waveform_dict[idx] = waveforms
        stft = librosa.stft(np.array(waveforms),n_fft=1024,hop_length=256,win_length=1024)[1:]
        #print("STFT", stft.shape,len(waveforms)/22500,sum(durations))
        stft_abs = np.abs(stft)
        stft_dict[idx] = stft_abs
        np.save(ronsen_path,
                stft_abs, allow_pickle=False)
        if finish_count%100==0:
            print("finish loop",finish_count,idx)
        finish_count += 1


        #for n in range(top_n):
        #    idx = text_list[n][1]["idx"]
        #    print("Idx",idx)
    print(not_in_file)
    print("Not in file count", len(not_in_file))
    print("finish")

    return stft_dict

    #print(waveform_dict["LJ001-0001"])


def get_data_to_buffer():
    buffer = list()
    text = process_text(os.path.join("data", hparams.train_file))
    a_time = time.time()
    stft_dict = get_prior_phoneme_sepctrogram_info(top_n=len(text))
    b_time = time.time()
    print("Load stft volume velocity time cost", b_time-a_time)
    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        mel_gt_name = os.path.join(
            hparams.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i + 1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            hparams.alignment_path, str(i) + ".npy"))
        tt = text[i].split("|")[1]
        idx = text[i].split("|")[0]
        character = tt[0:len(tt) - 1]
        character = np.array(
            text_to_sequence(character, hparams.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target, "idx": idx,"stft_volume_velocity":stft_dict[idx]})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end - start))

    return buffer


#vv = get_prior_phoneme_sepctrogram_info(top_n=1)
#print(vv["LJ001-0001"][2])
