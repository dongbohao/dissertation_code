import numpy as np
import data_function
from tacotron2_common.utils import to_device
from glottal_flow import get_torch_fft
import torch
import torch.nn as nn
import hparams as hp
import utils
from transformer.Models import Encoder, Decoder
from transformer.Layers import Linear, PostNet
from modules import LengthRegulator, CBHG
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import argparse
device = hp.device

# set seed
np.random.seed(61112)
torch.manual_seed(61112)
print(torch.__version__)
print("Cuda staus",torch.cuda.is_available())
print("Device",torch.cuda.get_device_name())
#device = torch.device('cpu')



class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self,encoder_n_layer,encoder_head,dropout):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(n_layers = encoder_n_layer, n_head = encoder_head,dropout = dropout)
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder(n_layers = encoder_n_layer, n_head = encoder_head,dropout = dropout)
        self.decoder1 = Decoder(n_layers = encoder_n_layer, n_head = encoder_head,dropout = dropout)


        self.mel_linear = Linear(hp.decoder_dim, hp.num_mels)

        self.pressuer_linear = Linear(512, 512)
        self.velocity_linear = Linear(512, 512)
        self.chainA_linear = Linear(512, 512)
        self.chainB_linear = Linear(512, 512)
        self.melout_linear = Linear(80, 80)

        self.rl1 = nn.LeakyReLU()
        self.rl2 = nn.LeakyReLU()
        self.rl3 = nn.LeakyReLU()
        self.rl4 = nn.LeakyReLU()


        self.l5 =  Linear(512, 512)
        self.l6 =  Linear(512, 512)
        self.l7 =  Linear(512, 512)
        self.l8 =  Linear(512, 512)


        self.rl5 = nn.LeakyReLU()
        self.rl6 = nn.LeakyReLU()
        self.rl7 = nn.LeakyReLU()
        self.rl8 = nn.LeakyReLU()
        self.mel_pressure = torchaudio.transforms.MelScale(n_stft=512,
                                                                     n_mels=80,
                                                                     f_max=8000.0).float()



        self.postnet = CBHG(hp.num_mels, K=8,
                            projections=[256, hp.num_mels])
        self.last_linear = Linear(hp.num_mels * 2, hp.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        #print("TTTT",lengths.device,mel_max_length.device)
        mask = ~utils.get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        encoder_output, _ = self.encoder(src_seq, src_pos)
        #print("Eecoder out",encoder_output.size())

        if self.training:
            length_regulator_output, duration_predictor_output = self.length_regulator(encoder_output,
                                                                                       target=length_target,
                                                                                       alpha=alpha,
                                                                                       mel_max_length=mel_max_length)

            #print("Length out", length_regulator_output.size(),duration_predictor_output.size())
            decoder_output = self.decoder(length_regulator_output, mel_pos)
            #print("Decoder out", decoder_output.size())
            #mel_output = self.mel_linear(decoder_output)

            #add chain matrix
            decoder_output1 = self.decoder1(length_regulator_output, mel_pos)
            #print("Decoder out 1",decoder_output1.size())


            stft_pressure = self.pressuer_linear(decoder_output[:,:,:512])
            stft_velocity = self.pressuer_linear(decoder_output[:, :, 512:1024])

            #print()
            chainA = self.chainA_linear(decoder_output1[:,:,:512])
            chainB = self.chainB_linear(decoder_output1[:,:,512:1024])

            stft_pressure = self.rl1(stft_pressure)
            stft_velocity = self.rl2(stft_velocity)

            chainA = self.rl3(chainA)
            chainB = self.rl4(chainB)

            stft_pressure = self.l5(stft_pressure)
            stft_velocity = self.l6(stft_velocity)
            chainA = self.l7(chainA)
            chainB = self.l8(chainB)

            stft_pressure = self.rl5(stft_pressure)
            stft_velocity = self.rl6(stft_velocity)
            chainA = self.rl7(chainA)
            chainB = self.rl8(chainB)


            stft_out = stft_pressure * chainA + stft_velocity * chainB

            mel_output = self.mel_pressure(stft_out.permute(0,2,1)).float()
            mel_output = self.melout_linear(mel_output.permute(0,2,1))



            mel_output = self.mask_tensor(mel_output, mel_pos, mel_max_length)
            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual
            mel_postnet_output = self.mask_tensor(mel_postnet_output,
                                                  mel_pos,
                                                  mel_max_length)

            return mel_output, mel_postnet_output, duration_predictor_output, stft_pressure,stft_velocity,chainA,chainB
        else:
            length_regulator_output, decoder_pos = self.length_regulator(encoder_output,
                                                                         alpha=alpha)

            decoder_output = self.decoder(length_regulator_output, decoder_pos)

            mel_output = self.mel_linear(decoder_output)
            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual

            return mel_output, mel_postnet_output




#### LJspeech load
from torch.utils.data import DataLoader

train_sampler = None
shuffle = True


class ScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, d_model, n_warmup_steps, current_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = current_steps
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr_frozen(self, learning_rate_frozen):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = learning_rate_frozen
        self._optimizer.step()

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def get_learning_rate(self):
        learning_rate = 0.0
        for param_group in self._optimizer.param_groups:
            learning_rate = param_group['lr']

        return learning_rate

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class Tacotron2Loss(nn.Module):

    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def weighted_mse_loss(self,input, target, weight):
        return torch.mean(weight * (input - target) ** 2)

    def forward(self, mel, mel_postnet, duration_predicted, mel_target, duration_predictor_target,
                volume_velocity_target,volume_velocity_predicted,matrix_A_target,matrix_A_predict,matrix_B_target,matrix_B_predict):
        mel_target.requires_grad = False
        mel_loss = self.mse_loss(mel, mel_target)
        #mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

        #duration_predictor_target.requires_grad = False
        #duration_predictor_loss = self.l1_loss(duration_predicted,
        #                                       duration_predictor_target.float())

        mel_postnet_loss = 0
        duration_predictor_loss = 0
        #volume_velocity_loss = 1*self.mse_loss(volume_velocity_predicted,volume_velocity_target)
        volume_velocity_loss = 0
        #matrix_A_loss = 1 * self.mse_loss(matrix_A_predict, matrix_A_target)
        matrix_A_loss = 0
        #matrix_B_loss = 1 * self.mse_loss(matrix_B_predict, matrix_B_target)
        matrix_B_loss = 0
        return mel_loss, mel_postnet_loss, duration_predictor_loss,volume_velocity_loss,matrix_A_loss,matrix_B_loss


#ntokens = len(vocab)  # size of vocabulary
ntokens = 300
print("NNNN",ntokens)
emsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
#model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, ag.n_mel_channels).cuda()

#num_param = utils.get_param_num(model)
#criterion = nn.MSELoss()
criterion = Tacotron2Loss()
lr = 0.01  # learning rate
#optimizer = torch.optim.Adam(model.parameters(),
#                                 betas=(0.9, 0.98),
#                                 eps=1e-9)
train_loss_list = []
val_loss_list = []

parser = argparse.ArgumentParser(description="aaaaa")
parser.add_argument("--sections",type=int, default=0)
args = parser.parse_args()
sec_name = args.sections
print("Current section",sec_name)
load_from_checkpoint = False
if load_from_checkpoint:
    print("Load model from checkpoint")
    checkpoint = torch.load(r"checkpoint_v9_vv_mm_%s.pt"%str(sec_name),map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss_list = checkpoint['loss']


#scheduled_optim = ScheduledOptim(optimizer,
#                                     hp.decoder_dim,
#                                     hp.n_warm_up_step,
#                                     0)

#num_param = utils.get_param_num(model)

log_interval = 10
import time

from data_function import get_data_to_buffer,BufferDataset,collate_fn_tensor
buffer_train = get_data_to_buffer("test_%s.txt"%str(sec_name))
dataset_train = BufferDataset(buffer_train)

training_loader = DataLoader(dataset_train,
                                 batch_size=hp.batch_expand_size * hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn_tensor,
                                 drop_last=True,
                                 num_workers=0)




tl = []

for i, batchs in enumerate(training_loader):
     for j, db in enumerate(batchs):
         tl.append(db)

def train(tloader):
    #model.train()
    ttl = 0.
    history_train_loss = 0.
    total_time = 0.
    vv_train_loss = 0
    a_train_loss = 0
    b_train_loss = 0

    print("Epoch size", len(tloader))
    for i, db in enumerate(tloader):
            a_time = time.time()
            character = db["text"].long().to(device)
            mel_target = db["mel_target"].float().to(device)
            duration = db["duration"].int().to(device)
            mel_pos = db["mel_pos"].long().to(device)
            src_pos = db["src_pos"].long().to(device)
            max_mel_len = db["mel_max_len"]
            #volume_velocity_target = get_torch_fft(mel_target.size(0), mel_target.size(1), 512).float().to(device)
            volume_velocity_target = db["stft_volume_velocity"].float().to(device)

            matrix_A_target = db["matrix_A"].float().to(device)
            matrix_B_target = db["matrix_B"].float().to(device)

            #print("character", character.size())
            #print("duration",duration.size())
            if not character.size(1)==duration.size(1):
                continue

            mel_output, mel_postnet_output, duration_predictor_output, stft_pressure,stft_velocity,chainA,chainB = model(character,
                                                                              src_pos,
                                                                              mel_pos=mel_pos,
                                                                              mel_max_length=max_mel_len,
                                                                              length_target=duration)

            # Cal Loss
            mel_loss, mel_postnet_loss, duration_loss, volume_velocity_loss,matrix_A_loss,matrix_B_loss = criterion(mel_output,
                                                                        mel_postnet_output,
                                                                        duration_predictor_output,
                                                                        mel_target,
                                                                        duration,
                                                                        volume_velocity_target,
                                                                        stft_velocity ,
                                                                        matrix_A_target,chainA,
                                                                        matrix_B_target,chainB)
            total_loss = mel_loss + mel_postnet_loss + duration_loss + volume_velocity_loss + matrix_A_loss + matrix_B_loss

            # Logger
            t_l = total_loss.item()
            m_l = mel_loss.item()
            #m_p_l = mel_postnet_loss.item()
            #d_l = duration_loss.item()

            m_p_l = 0
            d_l = 0
            v_v = 0
            ls = t_l

            ma_l = 0
            mb_l = 0

            total_loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), hp.grad_clip_thresh)
            #scheduled_optim.step_and_update_lr()
            scheduled_optim.step_and_update_lr_frozen(5e-05)



            b_time = time.time()
            ttl += m_l
            vv_train_loss += v_v
            a_train_loss += ma_l
            b_train_loss += mb_l
            history_train_loss += m_l
            total_time += (b_time - a_time)
            if i % log_interval == 0 and i > 0:
                print("mel loss",m_l,
                      "time cost",(b_time-a_time),
                      "current lr",scheduled_optim.get_learning_rate(),
                      "v_v_loss",v_v,
                      "ma_loss",ma_l,
                      "mb_loss",mb_l,
                      "sec_name",sec_name)
                ttl = 0
                total_time = 0



        #if i >=0:
        #    break
    return list(map(lambda x:x/(i+1),[history_train_loss,vv_train_loss,a_train_loss,b_train_loss]))






def valid(vloader):
    #model.train()
    ttl = 0.
    history_val_loss = 0.
    vv_val_loss = 0.
    a_val_loss = 0.
    b_val_loss = 0.

    total_time = 0.



    print("Epoch valid size", len(vloader))
    for i, db in enumerate(vloader):
            a_time = time.time()
            character = db["text"].long().to(device)
            mel_target = db["mel_target"].float().to(device)
            duration = db["duration"].int().to(device)
            mel_pos = db["mel_pos"].long().to(device)
            src_pos = db["src_pos"].long().to(device)
            max_mel_len = db["mel_max_len"]
            volume_velocity_target = db["stft_volume_velocity"].float().to(device)

            matrix_A_target = db["matrix_A"].float().to(device)
            matrix_B_target = db["matrix_B"].float().to(device)

            #print("character", character.size())
            #print("duration",duration.size())
            if not character.size(1)==duration.size(1):
                continue

            mel_output, mel_postnet_output, duration_predictor_output, stft_pressure,stft_velocity,chainA,chainB = model(character,
                                                                              src_pos,
                                                                              mel_pos=mel_pos,
                                                                              mel_max_length=max_mel_len,
                                                                              length_target=duration)

            # Cal Loss
            mel_loss, mel_postnet_loss, duration_loss, volume_velocity_loss,matrix_A_loss,matrix_B_loss = criterion(mel_output,
                                                                        mel_postnet_output,
                                                                        duration_predictor_output,
                                                                        mel_target,
                                                                        duration,
                                                                        volume_velocity_target,
                                                                        stft_velocity ,
                                                                        matrix_A_target,chainA,
                                                                        matrix_B_target,chainB)
            total_loss = mel_loss + mel_postnet_loss + duration_loss + volume_velocity_loss + matrix_A_loss + matrix_B_loss

            # Logger
            t_l = total_loss.item()
            m_l = mel_loss.item()
            #m_p_l = mel_postnet_loss.item()
            #d_l = duration_loss.item()

            m_p_l = 0
            d_l = 0
            v_v = 0
            ls = t_l

            ma_l = 0
            mb_l = 0


            b_time = time.time()
            ttl += m_l
            vv_val_loss += v_v
            a_val_loss += ma_l
            b_val_loss += mb_l
            history_val_loss += m_l
            total_time += (b_time - a_time)
            if (j*i+j) % log_interval == 0 and j > 0:
                print("mel loss",m_l,
                      "time cost",(b_time-a_time),
                      "current lr",scheduled_optim.get_learning_rate(),
                      "v_v_loss",v_v,
                      "ma_loss",ma_l,
                      "mb_loss",mb_l,
                      "sec_name",sec_name)
                ttl = 0
                total_time = 0



        #if i >=0:
        #    break
    return list(map(lambda x:x/(i+1),[history_val_loss,vv_val_loss,a_val_loss,b_val_loss]))





# validation
def val(tloader,vloader):
    print("Loader len",len(tloader),len(vloader))
    for e in range(0,hp.epochs):
        c_time = time.time()
        train_loss = train(tloader)
        d_time = time.time()
        val_loss = valid(vloader)
        e_time = time.time()

        val_loss_list.append(val_loss)
        train_loss_list.append(train_loss)

        print("Train loss ", train_loss)
        print("Val loss ", val_loss)
        print("Epoch time cost", (d_time - c_time))
        print("Val time cost",(e_time - d_time))
        print("Sec_res",sec_name)
        return val_loss






encoder_n_layers = [1,2,4]
encoder_heads = [1,2,3]
#encoder_heads=[1]
dropouts = [0.1,0.3,0.5]
#dropouts = [0.1]

ret_dict = []

for enl in encoder_n_layers:
    for eh in encoder_heads:
        for dt in dropouts:
            hp.encoder_n_layer = enl
            hp.encoder_head = eh
            hp.dropout = dt
            print("Current parameters",enl,eh,dt)
            vll = 0
            s1 = [tl[:40],tl[40:]]
            s2 = [tl[10:],tl[:10]]
            s3 = [tl[:20]+tl[30:],tl[20:30]]
            for s in [s1,s2,s3]:
                model = FastSpeech(enl,eh,dt).to(device)
                model = model.train()
                optimizer = torch.optim.Adam(model.parameters(),
                                 betas=(0.9, 0.98),
                                 eps=1e-9)
                scheduled_optim = ScheduledOptim(optimizer,
                                     hp.decoder_dim,
                                     hp.n_warm_up_step,
                                     0)
                val_l = val(s[0],s[1])
                vll += val_l[0]
            ret_dict.append({"val_loss":vll/3,"enl":enl,"eh":eh,"dt":dt})

print(ret_dict)

#test_loss = ttst()
#print("Test loss:",test_loss)
#print("Sec_res",sec_name)

#tt_dataset()












