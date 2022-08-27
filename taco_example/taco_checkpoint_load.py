import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from glottal_model_v9 import FastSpeech,optimizer,test_loader,print_spectrogram,criterion
import data_function
from tacotron2_common.utils import to_gpu,to_device
from ABCD import print_fre
import matplotlib.pyplot as plt
import numpy as np
from hparams import device

model = FastSpeech().cpu()
#model = FastSpeech().to(device)
model = model.train()

#checkpoint = torch.load(r"C:\Users\xelloss\Downloads\checkpoint_v5_10ep_lr1-3.pt")
checkpoint = torch.load(r"checkpoint_v9_vv_mm.pt",map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print("Training loss", loss)
#model.eval()
#model = model.to("cpu")

print("model device",next(model.parameters()).device)


def print_time_domain(stft,n_fft=1024,sr=22500,win_length=1024,hop_length =256,pic_name=""):
    stft = stft.to("cpu").squeeze().detach().numpy().T
    print(stft.shape) # 512  T_out

    y = librosa.istft(stft,n_fft=n_fft,hop_length=hop_length,win_length=win_length)
    print(y.shape)

    y = y[:70]


    ind = list(map(lambda x:x, np.arange(y.shape[0])))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

    ax1.plot(ind,y, label='Magnitude')
    #ax1.set_ylim([-0.006,0.006])
    #for i, (x, y) in enumerate(zip(ind, frequency_points)):
    #    ax1.text(x, y, '%.2f' % y, ha='center', va='bottom')


    #ax1.set_ylabel('magnitude')
    ax1.set_xlabel('samples')

    #x_ticks_positions = [n for n in range(0, n_fft // 2, n_fft // 16)]
    #x_ticks_labels = [str(sr / 512 * n) + 'Hz' for n in x_ticks_positions]
    #plt.xticks(x_ticks_positions,x_ticks_labels)
    # if matrix_name == "A" or matrix_name == "D":
    #     ax1.set_ylim(ymin = 1-6e-4,ymax = 1+6e-4)
    plt.title('')

    #plt.xticks(ind + width / 2, ('train_accuracy', 'val_accuracy', 'test_accuracy', 'training_time(seconds)'))
    plt.legend(loc='best')
    #plt.savefig("%s.png"%,dpi=300)
    #plt.show()
    plt.savefig("v9_vv_%s.png"%pic_name)


def print_fft(stft,pic_name=""):
    stft = stft.to("cpu").squeeze().detach().numpy().T
    stft = stft[:,:1].flatten()
    print(stft.shape)
    t = np.linspace(1,512,512)
    plt.clf()
    plt.plot(t, stft, label="volume velocity")
    plt.legend(loc='best')
    plt.xlabel("Frequence")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.savefig("v9_vv_%s.png" % pic_name)

from glottal_flow import get_torch_fft

def tt_dataset():
    for i, batchs in enumerate(test_loader):
        for j,db in enumerate(batchs):
            character = db["text"].long().to("cpu")
            mel_target = db["mel_target"].float().to("cpu")
            duration = db["duration"].int().to("cpu")
            mel_pos = db["mel_pos"].long().to("cpu")
            src_pos = db["src_pos"].long().to("cpu")
            max_mel_len = db["mel_max_len"]
            volume_velocity_target = db["stft_volume_velocity"].float().to(device)
            #volume_velocity_target = get_torch_fft(mel_target.size(0), mel_target.size(1), 512).float().to("cpu")

            matrix_A_target = db["matrix_A"].float().to(device)
            matrix_B_target = db["matrix_B"].float().to(device)

            print(mel_target.device)


            mel_output, mel_postnet_output, duration_predictor_output, stft_pressure,stft_velocity,chainA,chainB = model(character,
                                                                              src_pos,
                                                                              mel_pos=mel_pos,
                                                                              mel_max_length=max_mel_len,
                                                                              length_target=duration)

            # Cal Loss
            mel_loss, mel_postnet_loss, duration_loss,vvloss, ma_loss, mb_loss = criterion(mel_output,
                                                                  mel_postnet_output,
                                                                  duration_predictor_output,
                                                                  mel_target,
                                                                  duration,volume_velocity_target,stft_velocity,
                                                                                           matrix_A_target,chainA,
                                                                                           matrix_B_target,chainB)

            print("Test loss", mel_loss.item())
            mel_output = mel_output[:1,:,:]
            print("mel test",mel_output.size())
            mel_target = mel_target[:1, :, :]
            print_spectrogram(mel_output,pic_name="predict_%s"%i)
            #print_spectrogram(mel_target,pic_name="ground_truth_%s"%i)

            stft_pressure = stft_pressure[:1, :, :]
            print_spectrogram(stft_pressure, pic_name="g_pressure_%s" % i)
            print_spectrogram(torch.abs(stft_pressure), pic_name="g_pressure_abs_%s" % i)

            #print_spectrogram(stft_pressure, pic_name="g_pressure_%s"%i)
            stft_velocity = stft_velocity[:1,:,:]
            print_time_domain(stft_velocity,pic_name="velocity_timedomain_predict")
            print_fft(stft_velocity,pic_name = "velocity_fft_predict")

            volume_velocity_target = volume_velocity_target[:1, :, :]
            print_time_domain(volume_velocity_target, pic_name="velocity_timedomain_target")
            print_fft(volume_velocity_target,pic_name="velocity_fft_target")

            print_spectrogram(volume_velocity_target, pic_name="g_velocity_target_%s" % i)

            print_spectrogram(torch.abs(stft_velocity), pic_name="g_velocity_abs_%s" % i)


            chainA = chainA[:1, :, :]
            print_spectrogram(chainA, pic_name="matrix_A_%s" % i)
            print_spectrogram(torch.abs(chainA), pic_name="matrix_A_abs_%s" % i)
            chainB = chainB[:1, :, :]
            print_spectrogram(chainB, pic_name="matrix_B_%s" % i)
            print_spectrogram(torch.abs(chainB), pic_name="matrix_B_abs_%s" % i)






            if i>=0:
                break
        if i >= 0:
            break
tt_dataset()

