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
device = hp.device

# set seed
np.random.seed(61112)
torch.manual_seed(61112)
print(torch.__version__)
print("Cuda staus",torch.cuda.is_available())
print("Device",torch.cuda.get_device_name())
#device = torch.device('cpu')
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


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder()
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder()

        self.mel_linear = Linear(hp.decoder_dim, hp.num_mels)

        self.pressuer_linear = Linear(512, 512)
        self.velocity_linear = Linear(512, 512)
        self.chainA_linear = Linear(512, 512)
        self.chainB_linear = Linear(512, 512)
        self.melout_linear = Linear(80, 80)
        self.mel_pressure = torchaudio.transforms.MelScale(n_stft=512,
                                                                     n_mels=80,
                                                                     f_max=8000.0).float()
        self.decoder1 = Decoder()


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

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')
#vocab = build_vocab_from_iterator(map(tokenizer, text_iter(trainset_rawtext)), specials=['<unk>'])
#vocab.set_default_index(vocab['<unk>'])
vocab = None
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

    def forward(self, mel, mel_postnet, duration_predicted, mel_target, duration_predictor_target, volume_velocity_target,volume_velocity_predicted):
        mel_target.requires_grad = False
        mel_loss = self.mse_loss(mel, mel_target)
        #mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

        #duration_predictor_target.requires_grad = False
        #duration_predictor_loss = self.l1_loss(duration_predicted,
        #                                       duration_predictor_target.float())

        mel_postnet_loss = 0
        duration_predictor_loss = 0
        volume_velocity_loss = 500*self.mse_loss(volume_velocity_predicted,volume_velocity_target)

        return mel_loss, mel_postnet_loss, duration_predictor_loss,volume_velocity_loss


#ntokens = len(vocab)  # size of vocabulary
ntokens = 300
print("NNNN",ntokens)
emsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
#model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, ag.n_mel_channels).cuda()
model = to_device(FastSpeech())
model = model.train()

num_param = utils.get_param_num(model)
#criterion = nn.MSELoss()
criterion = Tacotron2Loss()
lr = 0.01  # learning rate
optimizer = torch.optim.Adam(model.parameters(),
                                 betas=(0.9, 0.98),
                                 eps=1e-9)
scheduled_optim = ScheduledOptim(optimizer,
                                     hp.decoder_dim,
                                     hp.n_warm_up_step,
                                     0)

num_param = utils.get_param_num(model)

log_interval = 10
import time

from data_function import get_data_to_buffer,BufferDataset,collate_fn_tensor
buffer = get_data_to_buffer()
dataset = BufferDataset(buffer)

training_loader = DataLoader(dataset,
                                 batch_size=hp.batch_expand_size * hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn_tensor,
                                 drop_last=True,
                                 num_workers=0)

def train():
    #model.train()
    ttl = 0.
    history_train_loss = 0.
    total_time = 0.



    print("Epoch size", len(training_loader) * hp.batch_expand_size)
    for i, batchs in enumerate(training_loader):
        for j, db in enumerate(batchs):
            a_time = time.time()
            character = db["text"].long().to(device)
            mel_target = db["mel_target"].float().to(device)
            duration = db["duration"].int().to(device)
            mel_pos = db["mel_pos"].long().to(device)
            src_pos = db["src_pos"].long().to(device)
            max_mel_len = db["mel_max_len"]
            #volume_velocity_target = get_torch_fft(mel_target.size(0), mel_target.size(1), 512).float().to(device)
            volume_velocity_target = db["stft_volume_velocity"].float().to(device)

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
            mel_loss, mel_postnet_loss, duration_loss, volume_velocity_loss = criterion(mel_output,
                                                                        mel_postnet_output,
                                                                        duration_predictor_output,
                                                                        mel_target,
                                                                        duration,
                                                                        volume_velocity_target,
                                                                        stft_velocity        )
            total_loss = mel_loss + mel_postnet_loss + duration_loss

            # Logger
            t_l = total_loss.item()
            m_l = mel_loss.item()
            #m_p_l = mel_postnet_loss.item()
            #d_l = duration_loss.item()

            m_p_l = 0
            d_l = 0
            v_v = volume_velocity_loss.item()
            ls = t_l

            total_loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), hp.grad_clip_thresh)
            #scheduled_optim.step_and_update_lr()
            scheduled_optim.step_and_update_lr_frozen(5e-05)



            b_time = time.time()
            ttl += m_l
            history_train_loss += m_l
            total_time += (b_time - a_time)
            if j % log_interval == 0 and j > 0:
                print("mel loss",m_l,"time cost",(b_time-a_time),"current lr",scheduled_optim.get_learning_rate(),"v_v_loss",v_v)
                ttl = 0
                total_time = 0



        #if i >=0:
        #    break
    return history_train_loss/hp.batch_expand_size/(i+1)

train_loss_list = []
val_loss_list = []


# validation
def val():
    for e in range(0,ag.epochs):
        c_time = time.time()
        train_loss = train()
        d_time = time.time()
        val_loss = 0.

        i = 1
        val_loss_list.append(val_loss/(i+1))
        train_loss_list.append(train_loss)

        print("Train loss list", train_loss_list)
        print("Val loss list", val_loss_list)
        print("Epoch time cost", (d_time - c_time))

        checkpoint_path = "checkpoint_v9_vv_0.pt"
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
        plt.savefig("v9_vv_%s.png"%pic_name)
    else:
        if ground_truth:
            plt.savefig("v9_vv_groun_truth.png")
        else:
            plt.savefig("v9_vv_test_predict.png")



test_loader = DataLoader(dataset,
                                 batch_size=hp.batch_expand_size * hp.batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn_tensor,
                                 drop_last=True,
                                 num_workers=0)

def tt_dataset():
    for i, batchs in enumerate(test_loader):
        for j,db in enumerate(batchs):
            character = db["text"].long().to(device)
            mel_target = db["mel_target"].float().to(device)
            duration = db["duration"].int().to(device)
            mel_pos = db["mel_pos"].long().to(device)
            src_pos = db["src_pos"].long().to(device)
            max_mel_len = db["mel_max_len"]
            volume_velocity_target = get_torch_fft(mel_target.size(0), mel_target.size(1), 512).float().to(device)

            mel_output, mel_postnet_output, duration_predictor_output, stft_pressure,stft_velocity,chainA,chainB = model(character,
                                                                              src_pos,
                                                                              mel_pos=mel_pos,
                                                                              mel_max_length=max_mel_len,
                                                                              length_target=duration)

            # Cal Loss
            mel_loss, mel_postnet_loss, duration_loss, volume_velocity_loss = criterion(mel_output,
                                                                  mel_postnet_output,
                                                                  duration_predictor_output,
                                                                  mel_target,
                                                                  duration,volume_velocity_target,stft_velocity)

            print("Test loss", mel_loss.item())
            mel_output = mel_output[:1,:,:]
            print("mel test",mel_output.size())
            mel_target = mel_target[:1, :, :]
            print_spectrogram(mel_output,pic_name="predict_%s"%i)
            print_spectrogram(mel_target,pic_name="ground_truth_%s"%i)

            stft_pressure = stft_pressure[:1,:,:]
            print_spectrogram(stft_pressure, pic_name="g_pressure_%s"%i)
            stft_velocity = stft_velocity[:1,:,:]
            print_spectrogram(stft_velocity, pic_name="g_velocity_%s"%i)

            print_spectrogram(torch.abs(stft_pressure), pic_name="g_pressure_abs_%s" % i)
            print_spectrogram(torch.abs(stft_velocity), pic_name="g_velocity_abs_%s" % i)

            chainA = chainA[:1,:,:]
            print_spectrogram(chainA, pic_name="matrix_A_%s"%i)
            chainB = chainB[:1,:,:]
            print_spectrogram(chainB, pic_name="matrix_B_%s"%i)






            if i>=0:
                break
        if i >= 0:
            break

val()
tt_dataset()












