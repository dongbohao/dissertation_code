import math
from typing import Tuple
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import dataset
import data_function
import torchaudio
from ABCD import make_chain_matrix
import torchmetrics
from tacotron2_common.layers import LinearNorm,ConvNorm
from tacotron2_common.utils import to_gpu, get_mask_from_lengths

from torch import save, load, no_grad, LongTensor

## https://pytorch.org/tutorials/beginner/transformer_tutorial.html
## https://jalammar.github.io/illustrated-transformer/



# set seed
np.random.seed(61112)
torch.manual_seed(61112)
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
        batch_size=5,
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


def get_zero_frame(tgt):
    B = tgt.size(0)
    d_hid = tgt.size(2)
    t_out = tgt.size(1)
    dtype = tgt.dtype
    device = tgt.device
    decoder_start_end = torch.zeros(
        B, t_out, d_hid,
        dtype=dtype, device=device)
    return decoder_start_end



bohaoDecoder_config = {
                       'max_decoder_steps': 2000,
                       'gate_threshold': 0.5,
                       'early_stopping': False}


class BohaoDecoder(nn.Module):
    def __init__(self,d_model, nhead, dropout,nlayers,hidden_dim,max_decoder_steps, gate_threshold,
                 early_stopping):
        super(BohaoDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.early_stopping = early_stopping

        decoder_layers = TransformerDecoderLayer(d_model, nhead, hidden_dim, dropout,batch_first=True)
        self.transformerdecoder = TransformerDecoder(decoder_layers, nlayers)

        self.fc_glottal_out = nn.Linear(hidden_dim,hidden_dim*2)
        print("GGGGate lyaer",hidden_dim)
        self.gate_layer = LinearNorm(
            hidden_dim, 1,
            bias=True, w_init_gain='sigmoid')

        self.sft = nn.Sigmoid()

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(
            B,1, self.hidden_dim,
            dtype=dtype, device=device)
        return decoder_input

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        RETURNS
        -------
        inputs: processed decoder inputs
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        #shape = (mel_outputs.shape[0], -1, self.n_mel_channels)
        #mel_outputs = mel_outputs.view(*shape)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        # mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs

    def decode(self, decoder_input, memory):

        #print("Bohaodecoder",decoder_input.size(),memory.size())
        hidden_output = self.transformerdecoder(tgt = decoder_input,memory=memory)
        glottal_output = self.fc_glottal_out(hidden_output)
        #print("hidd",hidden_output)
        gate_prediction = self.gate_layer(hidden_output)
        gate_prediction = self.sft(gate_prediction)
        #print("Gate predi", gate_prediction)
        gate_prediction = torch.squeeze(gate_prediction,dim=1)
        #print("FF111111111111111111111",glottal_output.size())
        #glottal_output = torch.permute(glottal_output,(1,0,2))
        #print("Bohadecoder FFFFFFFFFFFFFFFFFFFF", gate_prediction.size(),glottal_output.size())
        return hidden_output,glottal_output,gate_prediction

    def forward(self, memory, memory_lengths):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        mel_lengths = torch.zeros([memory.size(0)], dtype=torch.int32, device=memory.device)
        not_finished = torch.ones([memory.size(0)], dtype=torch.int32, device=memory.device)

        glottal_outputs, gate_outputs, alignments = (
            torch.zeros(1), torch.zeros(1), torch.zeros(1))
        first_iter = True
        c = 0
        while True:
            #decoder_input = self.prenet(decoder_input)
            hidden_output,glottal_output,gate_output = self.decode(decoder_input,memory)
            #print("Gate out",gate_output)
            if first_iter:
                glottal_outputs = glottal_output
                gate_outputs = gate_output.unsqueeze(1)
                first_iter = False
            else:
                glottal_outputs = torch.cat(
                    (glottal_outputs, glottal_output), dim=1)
                gate_outputs = torch.cat((gate_outputs, gate_output.unsqueeze(1)), dim=1)

            dec = torch.le(gate_output,
                           0.5).to(torch.int32).squeeze(1)
            #print("Not finish",gate_outputs.size(),glottal_outputs.size())
            not_finished = not_finished * dec
            mel_lengths += not_finished
            #print("att context", hidden_output)
            #print("Not finish", not_finished,gate_output,dec,"---",gate_output.size())
            if self.early_stopping and torch.sum(not_finished) == 0:
                print("stop by finished flag at loop", c)
                break
            if glottal_outputs.size(1) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            c += 1
            #print("current loop ", c)
            decoder_input = hidden_output

        glottal_output, gate_outputs = self.parse_decoder_outputs(
            glottal_output, gate_outputs)

        return glottal_outputs, gate_outputs



class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, channels: int = 80,batch_first: bool = True, ag=None):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, d_hid)
        self.pos_encoder = PositionalEncoding(d_hid, dropout)

        #self.decoder = nn.Embedding(channels, d_hid)
        #self.pos_decoder = PositionalEncoding(d_hid, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_glottal_source_encoder = TransformerEncoder(encoder_layers, nlayers)

        bohaoDecoder_config.update({"d_model":d_model,
                                    "nhead":nhead,
                                    "hidden_dim":d_hid,
                                    "nlayers":nlayers,
                                    "dropout":dropout})
        self.transformer_glottal_source_decoder = BohaoDecoder(**bohaoDecoder_config)

        #self.channel_dim = channels * 2

        self.n_fft_dim = int((0+ag.filter_length/2)*2)
        print("TTTT",self.n_fft_dim,d_hid)
        self.fc_glottal_out = nn.Linear(d_hid, self.n_fft_dim)

        self.ag = ag
        self.mic_loss = nn.Linear(self.ag.n_mel_channels, self.ag.n_mel_channels)
        self.log_mel = nn.Linear(self.ag.n_mel_channels, self.ag.n_mel_channels)

        chain_matrix = make_chain_matrix(sample_rate=ag.sampling_rate, n_fft=ag.filter_length, mel_channels=ag.n_mel_channels)
        self.chain_matrix_A = chain_matrix["A"][:,:,1:]
        self.chain_matrix_B = chain_matrix["B"][:,:,1:]

        self.mel_pressure = torchaudio.transforms.MelScale(n_stft=int(0 + self.ag.filter_length / 2),
                                                                         n_mels=self.ag.n_mel_channels,
                                                                         f_max=self.ag.mel_fmax).float()


    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)


    def forward(self, src):
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
        glottal_encoder_output = self.transformer_glottal_source_encoder(src)
        print("g encoder out", glottal_encoder_output.size())

        memory_lengths = torch.tensor([x.size(0) for x in glottal_encoder_output])
        print("mem len",memory_lengths)
        glottal_decoder_output, gate_outputs = self.transformer_glottal_source_decoder(glottal_encoder_output,memory_lengths)
        print("g decoder out", glottal_decoder_output.size())
        #glottal_output = self.fc_glottal_out(glottal_decoder_output)
        glottal_output = glottal_decoder_output
        print("g output", glottal_output.size())
        output_pressure = glottal_output[:,:,:512]
        output_velocity = glottal_output[:,:,512:]   # shape B,times,1+n_fft/2

        print("output pressure size", output_pressure.size())
        print("output velocity size", output_velocity.size())

        #output_pressure = output_pressure.permute(0,2,1)
        #output_velocity = output_velocity.permute(0,2,1)
        #stft_pressure = self.inversemel_pressure(output_pressure)
        #stft_velocity = self.inversemel_velocity(output_velocity)
        stft_pressure = output_pressure
        stft_velocity = output_velocity


        print("finish inverse mel",stft_pressure.size(),stft_velocity.size())
        #stft_velocity = stft_velocity.permute(0,2,1)  # shape B,times, 1+n_ttf/2
        #stft_pressure = stft_pressure.permute(0,2,1)
        chain_matrix_A = self.chain_matrix_A   # shape 1,1,1+n_ttf/2
        chain_matrix_B = self.chain_matrix_B

        #print("CCCC",chain_matrix_A.size())
        #print("PPPP stft",stft_pressure.size())

        stft_lips_output_pressure = chain_matrix_A * stft_pressure + chain_matrix_B * stft_velocity
        print("Stft after chain",stft_lips_output_pressure.size())
        mel_lips_output_pressure = self.mel_pressure(stft_lips_output_pressure.permute(0,2,1))
        mel_lips_output_pressure = mel_lips_output_pressure.permute(0,2,1)
        print("mel size",mel_lips_output_pressure.size())
        output_mel = self.mic_loss(mel_lips_output_pressure.float())
        output_mel = self.log_mel(output_mel)
        #print("Lips out put", mel_lips_output_pressure.size())
        return output_mel,gate_outputs


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



class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, mel_out, gate_out, mel_target,gate_target,):
        mel_target.requires_grad = True
        gate_target.requires_grad = True
        gate_target = gate_target.view(-1, 1)

        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target)
               #    + nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss



ntokens = len(vocab)  # size of vocabulary
print("NNNN",ntokens)
emsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
#model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, ag.n_mel_channels).to(device)
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, ag.n_mel_channels, batch_first=True, ag = ag)

#criterion = nn.MSELoss()
criterion = Tacotron2Loss()
lr = 0.01  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
log_interval = 10



def train():
    model.train()
    total_loss = 0.
    for i, batch in enumerate(train_loader):
        text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, len_x = batch
        #print("Gate padding",gate_padded)
        #print("AAAAA",text_padded.size())
        x, y, num_items = data_function.batch_to_gpu(batch)
        #print("XXXX0",x[0].size())
        target = y[0].permute(0,2,1) # y[0] is mel
        gate = y[1] # y[1] is gate padding
        #print("XXXX2", x[2].size(),target.size())
        #print("YYYYY",y[0].size())


        #target = torch.randn(1,1,80)

        target_fill_zero = torch.zeros(ag.batch_size, 2000 - target.size(1), target.size(2))
        target = torch.cat([target, target_fill_zero], dim=1)

        gate_fill_one = torch.zeros(ag.batch_size, 2000 - gate.size(1))
        gate = torch.cat([gate,gate_fill_one],dim=1)
        print("Fill zero target ", target.size())
        print("Fill ones gate",gate.size())

        pred_y_mel, pred_y_gate_output = model(x[0])

        print("gate", gate.size())
        print("pred gate",pred_y_gate_output.size())
        #pred_y_target = pred_y_mel.permute(0,2,1)
        pred_y_target = pred_y_mel
        print("Pred_y_mel", pred_y_mel.size())
        print("Pred_y_target", pred_y_target.size())
        print("Target size",target.size())
        print("Y",x[2].size())

        #print(gate)
        #print(pred_y_gate_output)


        #loss = criterion(pred_y_target,target.float())
        loss = criterion(pred_y_target, pred_y_gate_output ,target.float(),gate.float())
        #loss.requires_grad = True
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        print("Loss",loss.item())

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









