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
from tacotron2_common.utils import get_mask_from_lengths
from tacotron2_common.utils import to_gpu, to_device
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
        batch_size=2,
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



class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = to_device(ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1))
        self.location_dense = to_device(LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh'))

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim,
                 attention_dim, attention_location_n_filters,
                 attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = to_device(LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh'))
        self.memory_layer = to_device(LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh'))
        self.v = to_device(LinearNorm(attention_dim, 1, bias=False))
        self.location_layer = to_device(LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim))
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(2)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [to_device(LinearNorm(in_size, out_size, bias=False))
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, n_mel_channels, postnet_embedding_dim,
                 postnet_kernel_size, postnet_n_convolutions):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            to_device(nn.Sequential(
                to_device(ConvNorm(n_mel_channels, postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh')),
                to_device(nn.BatchNorm1d(postnet_embedding_dim))))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                to_device(nn.Sequential(
                    to_device(ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size, stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh')),
                    to_device(nn.BatchNorm1d(postnet_embedding_dim))))
            )

        self.convolutions.append(
            to_device(nn.Sequential(
                to_device(ConvNorm(postnet_embedding_dim, n_mel_channels,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear')),
                to_device(nn.BatchNorm1d(n_mel_channels))))
        )
        self.n_convs = len(self.convolutions)

    def forward(self, x):
        i = 0
        for conv in self.convolutions:
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
            i += 1

        return x



bohaoDecoder_config = {
                       #'mask_padding': False,
                       'n_mel_channels': 1024,
                       #'n_symbols': 148,
                       #'symbols_embedding_dim': 512,
                       #'encoder_kernel_size': 5,
                       #'encoder_n_convolutions': 3,
                       'encoder_embedding_dim':512,
                       'attention_rnn_dim': 1024,
                       'attention_dim': 128,
                       'attention_location_n_filters': 32,
                       'attention_location_kernel_size': 31,
                       'n_frames_per_step': 1,
                       'decoder_rnn_dim': 1024,
                       'prenet_dim': 256,
                       'max_decoder_steps': 2000,
                       'gate_threshold': 0.5,
                       'p_attention_dropout': 0.1,
                       'p_decoder_dropout': 0.1,
                       #'postnet_embedding_dim': 512,
                       #'postnet_kernel_size': 5,
                       #'postnet_n_convolutions': 5,
                       'early_stopping': False}


class BohaoDecoder(nn.Module):
    def __init__(self, n_mel_channels, n_frames_per_step,
                 encoder_embedding_dim, attention_dim,
                 attention_location_n_filters,
                 attention_location_kernel_size,
                 attention_rnn_dim, decoder_rnn_dim,
                 prenet_dim, max_decoder_steps, gate_threshold,
                 p_attention_dropout, p_decoder_dropout,
                 early_stopping):
        super(BohaoDecoder, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.early_stopping = early_stopping

        self.prenet = to_device(Prenet(
            n_mel_channels * n_frames_per_step,
            [prenet_dim, prenet_dim]))

        self.attention_rnn = to_device(nn.LSTMCell(
            prenet_dim + encoder_embedding_dim,
            attention_rnn_dim))

        self.attention_layer = to_device(Attention(
            attention_rnn_dim, encoder_embedding_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size))

        self.decoder_rnn = to_device(nn.LSTMCell(
            attention_rnn_dim + encoder_embedding_dim,
            decoder_rnn_dim, 1))

        self.linear_projection = to_device(LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim,
            n_mel_channels * n_frames_per_step))

        self.gate_layer = to_device(LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid'))

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
            B, self.n_mel_channels * self.n_frames_per_step,
            dtype=dtype, device=device)
        return decoder_input

    def initialize_decoder_states(self, memory):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device)
        attention_cell = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device)

        decoder_hidden = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device)
        decoder_cell = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device)

        attention_weights = torch.zeros(
            B, MAX_TIME, dtype=dtype, device=device)
        attention_weights_cum = torch.zeros(
            B, MAX_TIME, dtype=dtype, device=device)
        attention_context = torch.zeros(
            B, self.encoder_embedding_dim, dtype=dtype, device=device)

        processed_memory = self.attention_layer.memory_layer(memory)

        return (attention_hidden, attention_cell, decoder_hidden,
                decoder_cell, attention_weights, attention_weights_cum,
                attention_context, processed_memory)

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

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
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
        alignments = alignments.transpose(0, 1).contiguous()
        # (T_out, B) -> (B, T_out)
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        shape = (mel_outputs.shape[0], -1, self.n_mel_channels)
        mel_outputs = mel_outputs.view(*shape)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        # mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, attention_hidden, attention_cell,
               decoder_hidden, decoder_cell, attention_weights,
               attention_weights_cum, attention_context, memory,
               processed_memory, mask):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, attention_context), -1)

        attention_hidden, attention_cell = self.attention_rnn(
            cell_input, (attention_hidden, attention_cell))
        attention_hidden = F.dropout(
            attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (attention_weights.unsqueeze(1),
             attention_weights_cum.unsqueeze(1)), dim=1)
        attention_context, attention_weights = self.attention_layer(
            attention_hidden, memory, processed_memory,
            attention_weights_cat, mask)

        attention_weights_cum += attention_weights
        decoder_input = torch.cat(
            (attention_hidden, attention_context), -1)

        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input, (decoder_hidden, decoder_cell))
        decoder_hidden = F.dropout(
            decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (decoder_hidden, attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        #print("hidd context", decoder_hidden_attention_context)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (decoder_output, gate_prediction, attention_hidden,
                attention_cell, decoder_hidden, decoder_cell, attention_weights,
                attention_weights_cum, attention_context)

    def forward(self, memory, mel_l,memory_lengths):
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

        mask = get_mask_from_lengths(memory_lengths)
        (attention_hidden,
         attention_cell,
         decoder_hidden,
         decoder_cell,
         attention_weights,
         attention_weights_cum,
         attention_context,
         processed_memory) = self.initialize_decoder_states(memory)

        mel_lengths = torch.zeros([memory.size(0)], dtype=torch.int32, device=memory.device)
        not_finished = torch.ones([memory.size(0)], dtype=torch.int32, device=memory.device)

        mel_outputs, gate_outputs, alignments = (
            torch.zeros(1), torch.zeros(1), torch.zeros(1))
        first_iter = True
        c = 0
        for i in range(0,mel_l):
            decoder_input = self.prenet(decoder_input)
            (mel_output,
             gate_output,
             attention_hidden,
             attention_cell,
             decoder_hidden,
             decoder_cell,
             attention_weights,
             attention_weights_cum,
             attention_context) = self.decode(decoder_input,
                                              attention_hidden,
                                              attention_cell,
                                              decoder_hidden,
                                              decoder_cell,
                                              attention_weights,
                                              attention_weights_cum,
                                              attention_context,
                                              memory,
                                              processed_memory,
                                              mask)

            if first_iter:
                mel_outputs = mel_output.unsqueeze(0)
                gate_outputs = gate_output
                alignments = attention_weights
                first_iter = False
            else:
                mel_outputs = torch.cat(
                    (mel_outputs, mel_output.unsqueeze(0)), dim=0)
                gate_outputs = torch.cat((gate_outputs, gate_output), dim=0)
                alignments = torch.cat((alignments, attention_weights), dim=0)

            dec = torch.le(torch.sigmoid(gate_output),
                           self.gate_threshold).to(torch.int32).squeeze(1)


            not_finished = not_finished * dec
            mel_lengths += not_finished

            c += 1
            #print("current loop ", c)
            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments, mel_lengths



class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, channels: int = 80,batch_first: bool = True, ag=None):
        super(TransformerModel, self).__init__()
        self.encoder = to_device(nn.Embedding(ntoken, d_hid))
        self.pos_encoder = to_device(PositionalEncoding(d_hid, dropout))

        #self.decoder = nn.Embedding(channels, d_hid)
        #self.pos_decoder = PositionalEncoding(d_hid, dropout)

        encoder_layers = to_device(TransformerEncoderLayer(d_model, nhead, d_hid, dropout))
        self.transformer_glottal_source_encoder = to_device(TransformerEncoder(encoder_layers, nlayers))

        self.transformer_glottal_source_decoder = to_device(BohaoDecoder(**bohaoDecoder_config))

        #self.channel_dim = channels * 2

        self.n_fft_dim = int((0+ag.filter_length/2)*2)
        print("TTTT",self.n_fft_dim,d_hid)
        self.fc_glottal_out = nn.Linear(d_hid, self.n_fft_dim)


        self.ag = ag

        chain_matrix = make_chain_matrix(sample_rate=ag.sampling_rate, n_fft=ag.filter_length, mel_channels=ag.n_mel_channels)
        self.chain_matrix_A = to_gpu(chain_matrix["A"][:,:,1:])
        self.chain_matrix_B = to_gpu(chain_matrix["B"][:,:,1:])

        self.mel_pressure = to_device(torchaudio.transforms.MelScale(n_stft=int(0 + self.ag.filter_length / 2),
                                                                         n_mels=self.ag.n_mel_channels,
                                                                         f_max=self.ag.mel_fmax).float())


    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)


    def forward(self, src, mel_l):
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
        glottal_decoder_output, gate_outputs, alignments, mel_lengths = self.transformer_glottal_source_decoder(glottal_encoder_output,mel_l,memory_lengths)
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


        print("finish inverse mel")
        #stft_velocity = stft_velocity.permute(0,2,1)  # shape B,times, 1+n_ttf/2
        #stft_pressure = stft_pressure.permute(0,2,1)
        chain_matrix_A = self.chain_matrix_A   # shape 1,1,1+n_ttf/2
        chain_matrix_B = self.chain_matrix_B

        #print("CCCC",chain_matrix_A.size())
        #print("PPPP stft",stft_pressure.size())

        stft_lips_output_pressure = chain_matrix_A * stft_pressure + chain_matrix_B * stft_velocity
        #print("Stft after chain",stft_lips_output_pressure.size())
        mel_lips_output_pressure = self.mel_pressure(stft_lips_output_pressure.permute(0,2,1))

        #print("Lips out put", mel_lips_output_pressure.size())
        return mel_lips_output_pressure,gate_outputs


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

    def forward(self, mel_out, mel_target):
        mel_target.requires_grad = False
        mel_loss = nn.MSELoss()(mel_out, mel_target)
        return mel_loss



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

        x, y, num_items = data_function.batch_to_gpu(batch)
        #print("XXXX0",x[0].size())
        target = y[0].permute(0,2,1) # y[0] is mel
        gate = y[1] # y[1] is gate padding


        mel_length = to_gpu(torch.tensor([target.size(1)]))
        pred_y_mel, pred_y_gate_output = model(x[0],mel_length)

        print("gate", gate.size())
        print("pred gate",pred_y_gate_output.size())
        pred_y_target = pred_y_mel.permute(0,2,1)
        print("Pred_y_mel", pred_y_mel.size())
        print("Pred_y_target", pred_y_target.size())
        print("Target size",target.size())
        print("Y",x[2].size())

        print(gate)
        print(pred_y_gate_output)


        #loss = criterion(pred_y_target,target.float())
        loss = criterion(pred_y_target,target.float())
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



train()
print("Taco Trainset",trainset)









