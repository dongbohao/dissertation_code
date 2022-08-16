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
        epochs=3,
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

        self.upsample_glottal = torch.nn.ConvTranspose1d(hidden_dim,
                                                 hidden_dim*2,
                                                 128, stride=48)

        self.upsample_matrix = torch.nn.ConvTranspose1d(hidden_dim,
                                                         hidden_dim*2,
                                                         128, stride=48)


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

        memory = memory.permute(0,2,1)
        #print("Src ",memory.size())
        glottal_outputs = self.upsample_glottal(memory)
        #print("glottal size", glottal_outputs.size())
        glottal_outputs = glottal_outputs[:,:,:mel_l]
        #print("glottal size",glottal_outputs.size())
        chain_matrix = self.upsample_matrix(memory)
        chain_matrix = chain_matrix[:,:,:mel_l]
        #print("chain matrix", chain_matrix.size())



        return glottal_outputs,chain_matrix



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

        glottal_decoder_output,chain_matrix = self.transformer_glottal_source_decoder(glottal_encoder_output,mel_length,src_mask=src_mask)

        chain_matrix_A = chain_matrix[:, :512:].permute(0,2,1)      # B, T_out, channels
        chain_matrix_B = chain_matrix[:, 512:1024:].permute(0,2,1)
        #chain_matrix_C = chain_matrix[:, 1024:1536:].permute(0,2,1)
        #chain_matrix_D = chain_matrix[:, 1536:2048:].permute(0,2,1)

        stft_pressure = glottal_decoder_output[:,:512,:].permute(0,2,1)
        stft_velocity = glottal_decoder_output[:,512:1024,:].permute(0,2,1)

        stft_lips_output_pressure = chain_matrix_A * stft_pressure + chain_matrix_B * stft_velocity

        mel_lips_output_pressure = self.mel_pressure(stft_lips_output_pressure.permute(0,2,1))
        mel_lips_output_pressure = mel_lips_output_pressure.permute(0,2,1)

        output_mel = self.mic_loss(mel_lips_output_pressure.float())
        output_mel = self.log_mel(output_mel)

        return output_mel,stft_pressure, stft_velocity, chain_matrix_A, chain_matrix_B


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

testset_loader = DataLoader(testset, num_workers=0, shuffle=False,
                              sampler=train_sampler,
                              batch_size=1, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn_test)






import librosa
import librosa.display
import matplotlib.pyplot as plt




class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, mel_out, mel_target,):
        mel_target.requires_grad = True
        #gate_target.requires_grad = True

        #gate_target = gate_target.view(-1, 1)
        #gate_out = gate_out.view(-1, 1)

        mel_loss = nn.MSELoss()(mel_out, mel_target)
        #gate_loss = nn.BCELoss(reduction='mean')(gate_out, gate_target)
        return mel_loss



ntokens = len(vocab)  # size of vocabulary
print("NNNN",ntokens)
emsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
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
log_interval = 10



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
        #print("Target ",target.size())

        #src_mask = to_gpu(torch.not_equal(x[0],0).unsqueeze(2))
        src_mask = None

        mel_length = to_gpu(torch.tensor([target.size(1)]))

        pred_y_mel, predict_velocity, predict_pressure, predict_matrixA, predict_matrixB = model(x[0],mel_length,src_mask = src_mask)


        pred_y_target = pred_y_mel

        #print("pred ",pred_y_target.size())
        loss = criterion(pred_y_target,target.float())

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

        checkpoint_path = "checkpoint_v8.pt"
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss_list,
        }, checkpoint_path)
        print("checkpoint saved")







def print_spectrogram(pred_y_mel,ground_truth = False,pic_name = ""):
    pred_y_mel = pred_y_mel.to("cpu").squeeze().detach().numpy().T
    #log_spectro = librosa.amplitude_to_db(pred_y_mel)
    log_spectro = pred_y_mel
    print("mel",pred_y_mel.shape)

    #log_spectro = log_spectro[:,:gate_index]

    # Plotting the short-time Fourier Transformation
    plt.figure(figsize=(20, 5))
    # Using librosa.display.specshow() to create our spectrogram
    librosa.display.specshow(log_spectro, sr=ag.sampling_rate, x_axis='time', y_axis='hz', hop_length=ag.hop_length,
                             cmap='magma', fmax=80)
    plt.colorbar(label='Decibels')

    if pic_name:
        plt.savefig("v8_%s.png"%pic_name)
    else:
        if ground_truth:
            plt.savefig("v8_groun_truth.png")
        else:
            plt.savefig("v8_test_predict.png")



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
        pred_y_mel, predict_velocity, predict_pressure, predict_matrixA, predict_matrixB = model(x[0],mel_length)

        loss = criterion(pred_y_mel, target.float())
        print("Test loss", loss.item())

        print_spectrogram(pred_y_mel,pic_name="predict_%s"%i)
        print_spectrogram(target,pic_name="ground_truth_%s"%i)
        print_spectrogram(predict_velocity, pic_name="g_velocity_%s"%i)
        print_spectrogram(predict_pressure, pic_name="g_pressure_%s"%i)
        print_spectrogram(predict_matrixA, pic_name="matrix_A_%s"%i)
        print_spectrogram(predict_matrixB, pic_name="matrix_B_%s"%i)






        if i>0:
            break

val()
tt_dataset()












