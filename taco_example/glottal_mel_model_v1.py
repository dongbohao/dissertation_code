import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import dataset
import data_function
import torchaudio
from taco_example.ABCD import make_chain_matrix

from torch import save, load, no_grad, LongTensor

## https://pytorch.org/tutorials/beginner/transformer_tutorial.html
## https://jalammar.github.io/illustrated-transformer/

print(torch.__version__)

class Ag():
    def __init__(self,**kargs):
        self.kargs = kargs

    def __getattr__(self, item):
        return self.kargs[item]



ag = Ag(output='out.txt',
        dataset_path=r'D:\tacotron2\DeepLearningExamples\PyTorch\SpeechSynthesis\Tacotron2',
        model_name='Tacotron2',
        log_file='nvlog.json',
        anneal_steps=None,
        anneal_factor=0.1,
        config_file=None,
        seed=None,
        epochs=10,
        epochs_per_checkpoint=50,
        checkpoint_path='',
        resume_from_last=False,
        dynamic_loss_scaling=True,
        amp=True,
        cudnn_enabled=True,
        cudnn_benchmark=False,
        disable_uniform_initialize_bn_weight=False,
        use_saved_learning_rate=False,
        learning_rate=0.001,
        weight_decay=1e-06,
        grad_clip_thresh=1.0,
        batch_size=1,
        grad_clip=5.0,
        load_mel_from_disk=True,
        training_files='filelists/ljs_mel_text_train_subset_2500_filelist.txt',
        validation_files='filelists/ljs_mel_text_val_filelist.txt',
        text_cleaners=['english_cleaners'],
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        rank=0,
        world_size=1,
        dist_url='tcp://localhost:23456',
        group_name='group_name',
        dist_backend='nccl',
        bench_class='',
        n_mel_channels=80,
        n_frames_per_step=3,
        )










class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, channels: int = 80,batch_first: bool = True, ag=None):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, d_hid)
        self.pos_encoder = PositionalEncoding(d_hid, dropout)

        #self.decoder = nn.Embedding(channels, d_hid)
        #self.pos_decoder = PositionalEncoding(d_hid, dropout)


        self.transformer_glottal_pressure = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=nlayers,
                                          num_decoder_layers=nlayers, dim_feedforward=d_hid, dropout=dropout,batch_first=batch_first)

        self.transformer_glottal_velocity = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=nlayers,
                                                           num_decoder_layers=nlayers, dim_feedforward=d_hid,
                                                           dropout=dropout, batch_first=batch_first)

        self.fc_pressure_out = nn.Linear(d_hid, channels)
        self.fc_velocity_out = nn.Linear(d_hid, channels)

        self.ag = ag
        self.inversemel_pressure = torchaudio.transforms.InverseMelScale(n_stft=int(1+self.ag.filter_length/2), n_mels = self.ag.n_mel_channels,f_max = self.ag.mel_fmax,max_iter=10)
        self.inversemel_velocity = torchaudio.transforms.InverseMelScale(n_stft=int(1+self.ag.filter_length/2), n_mels = self.ag.n_mel_channels,f_max = self.ag.mel_fmax,max_iter=10)


        chain_matrix = make_chain_matrix(sample_rate=ag.sampling_rate, n_fft=ag.filter_length, mel_channels=ag.n_mel_channels)
        self.chain_matrix_A = chain_matrix["A"]
        self.chain_matrix_B = chain_matrix["B"]

        self.mel_pressure = torchaudio.transforms.MelScale(n_stft=int(1 + self.ag.filter_length / 2),
                                                                         n_mels=self.ag.n_mel_channels,
                                                                         f_max=self.ag.mel_fmax)





    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)


    def forward(self, src, trg):
        #if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
        #    self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        #src_pad_mask = self.make_len_mask(src)
        #trg_pad_mask = self.make_len_mask(trg)

        #print("SCR shape", src.size())
        #print("Trg shape", trg.size())

        src = self.encoder(src)
        src = self.pos_encoder(src)

        #trg = self.decoder(trg)
        #trg = self.pos_decoder(trg)

        #print("Scr",src.size())
        #print("Tgt",trg.size())
        output_pressure = self.transformer_glottal_pressure(src, trg)
        output_pressure = self.fc_pressure_out(output_pressure)

        output_velocity = self.transformer_glottal_velocity(src, trg)
        output_velocity = self.fc_velocity_out(output_velocity)   # shape 1,times,mel

        output_pressure = output_pressure.permute(0,2,1)
        output_velocity = output_velocity.permute(0,2,1)
        stft_pressure = self.inversemel_pressure(output_pressure)
        stft_velocity = self.inversemel_velocity(output_velocity)

        stft_velocity = stft_velocity.permute(0,2,1)  # shape 1,times, 1+n_ttf/2
        stft_pressure = stft_pressure.permute(0,2,1)
        chain_matrix_A = self.chain_matrix_A   # shape 1,1,1+n_ttf/2
        chain_matrix_B = self.chain_matrix_B

        #print("CCCC",chain_matrix_A.size())
        #print("PPPP stft",stft_pressure.size())

        stft_lips_output_pressure = chain_matrix_A * stft_pressure + chain_matrix_B * stft_velocity
        #print("Stft after chain",stft_lips_output_pressure.size())
        mel_lips_output_pressure = self.mel_pressure(stft_lips_output_pressure.permute(0,2,1))

        #print("Lips out put", mel_lips_output_pressure.size())
        return mel_lips_output_pressure


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)





class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




#### LJspeech load
from torch.utils.data import DataLoader

train_sampler = None
shuffle = True

trainset_rawtext = data_function.get_data_loader(
        ag.model_name, ag.dataset_path, ag.training_files, ag)

def text_iter(dataset):
    for item in dataset:
        yield item[0]



from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, text_iter(trainset_rawtext)), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

trainset = data_function.TextMelLoader(ag.dataset_path, ag.training_files, ag, vocab = vocab, tokenizer=tokenizer)

collate_fn = data_function.get_collate_function(
        ag.model_name, ag.n_frames_per_step)
valset = data_function.TextMelLoader(ag.dataset_path, ag.validation_files, ag, vocab = vocab, tokenizer=tokenizer)
train_loader = DataLoader(trainset, num_workers=0, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=ag.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)





# for a in trainset:
#     print("AAAAAA0", a[0])
#     print("AAAAAA1", a[1].size())
#     print("AAAAAA2", a[2])
#     print("AAAAAA3", len(a))

import librosa
import librosa.display
import matplotlib.pyplot as plt
import copy

ntokens = len(vocab)  # size of vocabulary
print("NNNN",ntokens)
emsize = 80  # embedding dimension
d_hid = 80  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
#model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, ag.n_mel_channels).to(device)
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, ag.n_mel_channels, batch_first=True, ag = ag)

criterion = nn.MSELoss()
lr = 1.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
log_interval = 10

def train():
    model.train()
    total_loss = 0.
    for i, batch in enumerate(train_loader):
        text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, len_x = batch
        #print("AAAAA",text_padded.size())
        x, y, num_items = data_function.batch_to_gpu(batch)
        #print("XXXX0",x[0].size())
        target = x[2].permute(0,2,1) # x[2] is mel
        #print("XXXX2", x[2].size(),target.size())
        #print("YYYYY",y[0].size())

        target = torch.randn(1,1,80)
        pred_y_mel = model(x[0], target)

        pred_y_target = pred_y_mel.permute(0,2,1)
        print("Pred_y_mel", pred_y_mel.size())
        print("Pred_y_target", pred_y_mel.size())
        print("Y",x[2].size())

        loss = criterion(target, pred_y_target)
        loss.requires_grad = True
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        print(loss.item())

        total_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'lr {lr:02.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0

        # log_spectro = librosa.amplitude_to_db(pred_y_mel.squeeze())
        # # Plotting the short-time Fourier Transformation
        # plt.figure(figsize=(20, 5))
        # # Using librosa.display.specshow() to create our spectrogram
        # librosa.display.specshow(log_spectro, sr=ag.sampling_rate, x_axis='time', y_axis='hz', hop_length=ag.hop_length, cmap='magma',
        #                          fmax=80)
        # plt.colorbar(label='Decibels')
        # plt.show()

        #break

train()
print("Taco Trainset",trainset)









