import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from glottal_mel_model_v5 import TfModel,optimizer,print_spectrogram,criterion,print_spectrogram,testset_loader,model
import data_function
from tacotron2_common.utils import to_gpu,to_device
from ABCD import print_fre
import matplotlib.pyplot as plt
import numpy as np
device = torch.device('cpu')


#checkpoint = torch.load(r"C:\Users\xelloss\Downloads\checkpoint_v5_10ep_lr1-3.pt")
checkpoint = torch.load(r"checkpoint_v5.pt",map_location=device)
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
    plt.savefig("v5_%s.png"%pic_name)


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
    plt.savefig("v5_%s.png" % pic_name)

from glottal_flow import get_torch_fft

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
        print_spectrogram(predict_velocity, gate, pic_name="g_velocity")
        print_spectrogram(predict_pressure, gate, pic_name="g_pressure")
        print_spectrogram(predict_matrixA, gate, pic_name="matrix_A")
        print_spectrogram(predict_matrixB, gate, pic_name="matrix_B")

        break
tt_dataset()

