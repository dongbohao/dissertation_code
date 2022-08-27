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
from tacotron2_common.utils import to_gpu, get_mask_from_lengths, to_device

from torch import save, load, no_grad, LongTensor

## https://pytorch.org/tutorials/beginner/transformer_tutorial.html
## https://jalammar.github.io/illustrated-transformer/



# set seed
np.random.seed(61112)
torch.manual_seed(61112)
print(torch.__version__)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Cuda staus",torch.cuda.is_available())
print("Device",torch.cuda.get_device_name())

class Ag():
    def __init__(self,**kargs):
        self.kargs = kargs

    def __getattr__(self, item):
        return self.kargs[item]



ag = Ag(output='out.txt',
        dataset_path=r'D:\tacotron2\DeepLearningExamples\PyTorch\SpeechSynthesis\Tacotron2',
        #dataset_path  = "/data/acq21bd/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2",
        model_name='Tacotron2',
        log_file='nvlog.json',
        anneal_steps=None,
        anneal_factor=0.1,
        config_file=None,
        seed=None,
        epochs=1,
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
        test_files='filelists/ljs_mel_text_test_filelist.txt',
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
        n_frames_per_step=1,
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
                 early_stopping, fc_out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.early_stopping = early_stopping

        decoder_layers = to_device(TransformerDecoderLayer(d_model, nhead, hidden_dim, dropout,batch_first=True))
        self.transformerdecoder = to_device(TransformerDecoder(decoder_layers, nlayers))

        self.fc_glottal_out = to_device(nn.Linear(hidden_dim,fc_out_dim))
        print("GGGGate lyaer",hidden_dim)
        self.gate_layer = to_device(LinearNorm(
            hidden_dim, 1,
            bias=True, w_init_gain='sigmoid'))

        self.sft = to_device(nn.Sigmoid())

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

    def decode(self, decoder_input, memory, src_mask = None):


        hidden_output = self.transformerdecoder(tgt = decoder_input,memory=memory,memory_mask = src_mask)
        glottal_output = self.fc_glottal_out(hidden_output)

        gate_prediction = self.gate_layer(hidden_output)
        gate_prediction = self.sft(gate_prediction)

        gate_prediction = torch.squeeze(gate_prediction,dim=1)

        return hidden_output,glottal_output,gate_prediction

    def forward(self, memory, mel_l, src_mask = None):
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
        #while True:
        for cont_index in range(0,mel_l):
            #decoder_input = self.prenet(decoder_input)
            hidden_output,glottal_output,gate_output = self.decode(decoder_input,memory,src_mask=src_mask)

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

            not_finished = not_finished * dec
            mel_lengths += not_finished



            # if self.early_stopping and torch.sum(not_finished) == 0:
            #     print("stop by finished flag at loop", c)
            #     break
            # if glottal_outputs.size(1) == self.max_decoder_steps:
            #     print("Warning! Reached max decoder steps")
            #     break

            c += 1

            decoder_input = hidden_output

        glottal_output, gate_outputs = self.parse_decoder_outputs(
            glottal_output, gate_outputs)

        return glottal_outputs, gate_outputs



class TfModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, channels: int = 80,batch_first: bool = True, ag=None):
        super().__init__()
        self.encoder = to_device(nn.Embedding(ntoken, d_hid))
        self.pos_encoder = to_device(PositionalEncoding(d_hid, dropout))

        #self.decoder = nn.Embedding(channels, d_hid)
        #self.pos_decoder = PositionalEncoding(d_hid, dropout)

        encoder_layers = to_device(TransformerEncoderLayer(d_model, nhead, d_hid, dropout))
        self.transformer_glottal_source_encoder = to_device(TransformerEncoder(encoder_layers, nlayers))

        self.n_fft_dim = int((0 + ag.filter_length / 2) * 4)
        print("TTTT", self.n_fft_dim, d_hid)
        bohaoDecoder_config.update({"d_model":d_model,
                                    "nhead":nhead,
                                    "hidden_dim":d_hid,
                                    "nlayers":nlayers,
                                    "dropout":dropout,
                                    "fc_out_dim":self.n_fft_dim})
        self.transformer_glottal_source_decoder = to_device(BohaoDecoder(**bohaoDecoder_config))

        self.ag = ag
        self.mic_loss = to_device(nn.Linear(self.ag.n_mel_channels, self.ag.n_mel_channels))
        self.log_mel = to_device(nn.Linear(self.ag.n_mel_channels, self.ag.n_mel_channels))

        chain_matrix = make_chain_matrix(sample_rate=ag.sampling_rate, n_fft=ag.filter_length, mel_channels=ag.n_mel_channels)

        #self.chain_matrix_A = chain_matrix["A"][:,:,1:]
        #self.chain_matrix_B = chain_matrix["B"][:,:,1:]

        self.mel_pressure = to_device(torchaudio.transforms.MelScale(n_stft=int(0 + self.ag.filter_length / 2),
                                                                         n_mels=self.ag.n_mel_channels,
                                                                         f_max=self.ag.mel_fmax)).float()


    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)


    def forward(self, src,mel_length, src_mask = None):

        src = self.encoder(src)
        src = self.pos_encoder(src)

        glottal_encoder_output = self.transformer_glottal_source_encoder(src,mask=src_mask)


        memory_lengths = torch.tensor([x.size(0) for x in glottal_encoder_output])

        glottal_decoder_output, gate_outputs = self.transformer_glottal_source_decoder(glottal_encoder_output,mel_length,src_mask=src_mask)

        glottal_output = glottal_decoder_output

        output_pressure = glottal_output[:,:,:512]
        output_velocity = glottal_output[:,:,512:1024]   # shape B,times,1+n_fft/2



        stft_pressure = output_pressure
        stft_velocity = output_velocity



        chain_matrix_A = glottal_output[:,:,1024:1536]   # shape B,1,0+n_ttf/2
        chain_matrix_B = glottal_output[:,:,1536:2048]



        stft_lips_output_pressure = chain_matrix_A * stft_pressure + chain_matrix_B * stft_velocity

        mel_lips_output_pressure = self.mel_pressure(stft_lips_output_pressure.permute(0,2,1))
        mel_lips_output_pressure = mel_lips_output_pressure.permute(0,2,1)

        output_mel = self.mic_loss(mel_lips_output_pressure.float())
        output_mel = self.log_mel(output_mel)

        return output_mel,gate_outputs,output_velocity, output_pressure, chain_matrix_A, chain_matrix_B


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




#### LJspeech load
from torch.utils.data import DataLoader

train_sampler = None
shuffle = True

trainset_rawtext = data_function.get_data_loader(
        ag.model_name, ag.dataset_path, ag.training_files, ag)

def text_iter(dataset):
    for item in dataset:
        #print(item)
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

collate_fn_val = data_function.get_collate_function(
        ag.model_name, ag.n_frames_per_step)

collate_fn_test = data_function.get_collate_function(
        ag.model_name, ag.n_frames_per_step)

testset = data_function.TextMelLoader(ag.dataset_path, ag.test_files, ag, vocab = vocab, tokenizer=tokenizer)

testset_loader = DataLoader(trainset, num_workers=0, shuffle=False,
                              sampler=train_sampler,
                              batch_size=1, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn_test)






import librosa
import librosa.display
import matplotlib.pyplot as plt




class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, mel_out, gate_out, mel_target,gate_target,):
        mel_target.requires_grad = True
        #gate_target.requires_grad = True

        #gate_target = gate_target.view(-1, 1)
        #gate_out = gate_out.view(-1, 1)

        mel_loss = nn.MSELoss()(mel_out, mel_target)
        #gate_loss = nn.BCELoss(reduction='mean')(gate_out, gate_target)
        return mel_loss



ntokens = len(vocab)  # size of vocabulary
print("NNNN",ntokens)
emsize = 128  # embedding dimension
d_hid = 128  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
#model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, ag.n_mel_channels).cuda()
model = TfModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, ag.n_mel_channels, batch_first=True, ag = ag)

#criterion = nn.MSELoss()
criterion = Tacotron2Loss()
lr = 0.01  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
log_interval = 2



def train():
    #model.train()
    total_loss = 0.
    history_train_loss = 0.
    train_loader = DataLoader(trainset, num_workers=0, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=ag.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    for i, batch in enumerate(train_loader):
        text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, len_x = batch

        x, y, num_items = data_function.batch_to_gpu(batch)

        target = y[0].permute(0,2,1) # y[0] is mel
        gate = y[1] # y[1] is gate padding

        #src_mask = to_gpu(torch.not_equal(x[0],0).unsqueeze(2))
        src_mask = None

        mel_length = to_gpu(torch.tensor([target.size(1)]))

        pred_y_mel, pred_y_gate_output, predict_velocity, predict_pressure, predict_matrixA, predict_matrixB = model(x[0],mel_length,src_mask = src_mask)


        pred_y_target = pred_y_mel

        loss = criterion(pred_y_target, pred_y_gate_output ,target.float(),gate.float())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


        total_loss += loss.item()
        history_train_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'lr {lr:02.2f} | '
                  f'train loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0



        #if i >0:
        #    break
    return history_train_loss/(i+1)

train_loss_list = []
val_loss_list = []


# validation
def val():
    for e in range(0,ag.epochs):
        train_loss = train()
        val_loss = 0.
        # valset_loader = DataLoader(valset, num_workers=0, shuffle=shuffle,
        #                            sampler=train_sampler,
        #                            batch_size=ag.batch_size, pin_memory=False,
        #                            drop_last=True, collate_fn=collate_fn_val)
        # for i, batch in enumerate(valset_loader):
        #     text_padded, input_lengths, mel_padded, gate_padded, \
        #     output_lengths, len_x = batch
        #     x, y, num_items = data_function.batch_to_gpu(batch)
        #     target = y[0].permute(0, 2, 1)  # y[0] is mel
        #     gate = y[1]  # y[1] is gate padding
        #     target_fill_zero = torch.zeros(ag.batch_size, 2000 - target.size(1), target.size(2))
        #     target_fill_zero = to_gpu(target_fill_zero).float()
        #     target = torch.cat([target, target_fill_zero], dim=1)
        #
        #     gate_fill_one = torch.zeros(ag.batch_size, 2000 - gate.size(1))
        #     gate_fill_one = to_gpu(gate_fill_one).float()
        #     gate = torch.cat([gate, gate_fill_one], dim=1)
        #
        #     pred_y_mel, pred_y_gate_output = model(x[0])
        #
        #     pred_y_target = pred_y_mel
        #
        #     #loss = criterion(pred_y_target, pred_y_gate_output, target.float(), gate.float())
        #     lossi = 0.1
        #     print("val loss", lossi)
        #
        #     val_loss += lossi
        #
        #     if i>1:
        #         break

        i = 1
        val_loss_list.append(val_loss/(i+1))
        train_loss_list.append(train_loss)

        print("Train loss list", train_loss_list)
        print("Val loss list", val_loss_list)

        checkpoint_path = "checkpoint_v5.pt"
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss_list,
        }, checkpoint_path)
        print("checkpoint saved")







def print_spectrogram(pred_y_mel,gate,ground_truth = False,pic_name = ""):
    pred_y_mel = pred_y_mel.to("cpu").squeeze().detach().numpy().T
    #log_spectro = librosa.amplitude_to_db(pred_y_mel)
    log_spectro = pred_y_mel
    print("mel",pred_y_mel.shape)
    gate_dec = torch.ge(gate,
                        0.5).to(torch.int32).squeeze().to("cpu")
    gate_list = list(x for x in gate_dec)
    if 1 in gate_list:
        gate_index = gate_list.index(1)
    else:
        gate_index = 2000
    #print("Gate index",gate_index)
    #log_spectro = log_spectro[:,:gate_index]

    # Plotting the short-time Fourier Transformation
    plt.figure(figsize=(20, 5))
    # Using librosa.display.specshow() to create our spectrogram
    librosa.display.specshow(log_spectro, sr=ag.sampling_rate, x_axis='time', y_axis='hz', hop_length=ag.hop_length,
                             cmap='magma', fmax=80)
    plt.colorbar(label='Decibels')

    if pic_name:
        plt.savefig("v5_%s.png"%pic_name)
    else:
        if ground_truth:
            plt.savefig("v5_groun_truth.png")
        else:
            plt.savefig("v5_test_predict.png")



def tt_dataset():
    for i, batch in enumerate(testset_loader):
        x, y, num_items = data_function.batch_to_gpu(batch)
        target = y[0].permute(0, 2, 1)  # y[0] is mel
        gate = y[1]  # y[1] is gate padding


        #target_fill_zero = torch.zeros(1, 2000 - target.size(1), target.size(2))
        #target_fill_zero = to_gpu(target_fill_zero).float()
        #target = torch.cat([target, target_fill_zero], dim=1)

        #gate_fill_one = torch.zeros(1, 2000 - gate.size(1))
        #gate_fill_one = to_gpu(gate_fill_one).float()
        #gate = torch.cat([gate, gate_fill_one], dim=1)

        mel_length = to_gpu(torch.tensor([target.size(1)]))
        pred_y_mel, pred_y_gate_output, predict_velocity, predict_pressure, predict_matrixA, predict_matrixB = model(x[0],mel_length)

        loss = criterion(pred_y_mel, pred_y_gate_output, target.float(), gate.float())
        print("Test loss", loss.item())

        print_spectrogram(pred_y_mel,pred_y_gate_output)
        print_spectrogram(target,gate,ground_truth=True)
        print_spectrogram(torch.abs(predict_velocity), gate, pic_name="g_velocity")
        print_spectrogram(torch.abs(predict_pressure), gate, pic_name="g_pressure")
        print_spectrogram(torch.abs(predict_matrixA), gate, pic_name="matrix_A")
        print_spectrogram(torch.abs(predict_matrixB), gate, pic_name="matrix_B")







        break

val()
tt_dataset()












