import os
# Mel
num_mels = 80
text_cleaners = ['english_cleaners']

# FastSpeech
vocab_size = 300
max_seq_len = 3000

encoder_dim = 1024
encoder_n_layer = 4
encoder_head = 2
encoder_conv1d_filter_size = 1024

decoder_dim = 1024
decoder_n_layer = 4
decoder_head = 2
decoder_conv1d_filter_size = 1024

fft_conv1d_kernel = (9, 1)
fft_conv1d_padding = (4, 0)

duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
dropout = 0.1

# Train
checkpoint_path = "./model_new"
logger_path = "./logger"
mel_ground_truth = "./mels"
alignment_path = "./alignments"

batch_size = 2
epochs = 2000
n_warm_up_step = 4000

learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0
decay_step = [500000, 1000000, 2000000]

save_step = 3000
log_step = 5
clear_Time = 20

batch_expand_size = 32


cmu_sli_path = os.path.join("data","plp.lmwtSIL_MELS.ctm")
cmu_phoneme_path = os.path.join("data", "phoneme.ctm")
cmu_phoneme_type_path = os.path.join("data", "phoneme_type.ctm")


phoneme_config = {"vowel":{"O_q":0.8,
                           "a_m":2/3,
                           "A":4/1000,
                           "F_0":125,
                           "type":"vowel",
                           "is_voiced":True,
                           "t_0":1/125,
                           "t_p":0.8*1/125 * 2/3 ,
                           "t_n":0.8*1/125 * 1/3},

                  "stop":{"O_q":0.8,
                           "a_m":2/3,
                           "A":4/1000,
                           "F_0":125,
                          "type":"stop",
                           "is_voiced":True,
                           "t_0":1/125,
                           "t_p":0.8*1/125 * 2/3 ,
                           "t_n":0.8*1/125 * 1/3},

                  "affricate":{"O_q":0.8,
                           "a_m":2/3,
                           "A":4/1000,
                           "F_0":125,
                            "type":"affricate",
                           "is_voiced":True,
                           "t_0":1/125,
                           "t_p":0.8*1/125 * 2/3 ,
                           "t_n":0.8*1/125 * 1/3},

                  "fricative":{"O_q":0.8,
                           "a_m":2/3,
                           "A":4/1000,
                           "F_0":125,
                            "type":"fricative",
                           "is_voiced":False,
                           "t_0":1/125,
                           "t_p":0.8*1/125 * 2/3 ,
                           "t_n":0.8*1/125 * 1/3},

                  "aspirate": {"O_q":0.8,
                           "a_m":2/3,
                           "A":4/1000,
                           "F_0":125,
                            "type":"aspirate",
                           "is_voiced":False,
                           "t_0":1/125,
                           "t_p":0.8*1/125 * 2/3 ,
                           "t_n":0.8*1/125 * 1/3},

                  "liquid":{"O_q":0.8,
                           "a_m":2/3,
                           "A":4/1000,
                           "F_0":125,
                            "type":"liquid",
                           "is_voiced":True,
                           "t_0":1/125,
                           "t_p":0.8*1/125 * 2/3 ,
                           "t_n":0.8*1/125 * 1/3},

                  "nasal":{"O_q":0.8,
                           "a_m":2/3,
                           "A":4/1000,
                           "F_0":125,
                           "type":"nasal",
                           "is_voiced":True,
                           "t_0":1/125,
                           "t_p":0.8*1/125 * 2/3 ,
                           "t_n":0.8*1/125 * 1/3},

                  "semivowel":{"O_q":0.8,
                           "a_m":2/3,
                           "A":4/1000,
                           "F_0":125,
                            "type":"semivowel",
                           "is_voiced":True,
                           "t_0":1/125,
                           "t_p":0.8*1/125 * 2/3 ,
                           "t_n":0.8*1/125 * 1/3},

                  "common":{"O_q":0.8,
                           "a_m":2/3,
                           "A":4/1000,
                           "F_0":125,
                            "type":"common",
                           "is_voiced":True,
                           "t_0":1/125,
                           "t_p":0.8*1/125 * 2/3 ,
                           "t_n":0.8*1/125 * 1/3},
                  }

train_file = "trainsub.txt"
#train_file = "train_mini.txt"

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
