import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def print_spectrogram(pred_y_mel,ylim = [],clim=[0,1],pic_name = "",sample_rate=22050,hop_length=256,version = "v9_vv",is_mel=False):
    pred_y_mel = pred_y_mel.to("cpu").squeeze().detach().numpy().T
    log_spectro = pred_y_mel
    #if is_mel:
    #    log_spectro = pred_y_mel
    #else:
    #    log_spectro = librosa.feature.melspectrogram(S=pred_y_mel,n_mels=80)

    print("mel",pred_y_mel.shape)
    plt.figure(figsize=(20, 5))
    #fig, ax = plt.subplots()

    librosa.display.specshow(log_spectro, sr=sample_rate, x_axis='time', y_axis='hz', hop_length=hop_length,
                             cmap='magma', fmax=512)
    plt.colorbar(label='Decibels')
    plt.clim(clim)
    #ax.set(title='Now with labeled axes!')
    if ylim:
        plt.ylim(ylim)
    if pic_name:
        if version:
            plt.savefig("%s_%s.png"%(version,pic_name))
            piccc_name = "%s_%s_hist.png"%(version,pic_name)
        else:
            plt.savefig("%s.png" % pic_name)
            piccc_name = "%s_hist.png" % pic_name

    print_hist(log_spectro,piccc_name)







def print_hist(data,pic_name):
    plt.clf()
    data = data.sum(1)
    #print(data.shape)
    data_list = data.tolist()
    aa = np.arange(len(data_list))
    #print(aa)
    plt.bar(aa,data_list,width=1)
    x_ticks_positions = [n for n in range(0, len(data_list)//1, len(data_list) // 8)]
    x_ticks_labels = [str(22050 / 512 * n) + 'Hz' for n in x_ticks_positions]
    plt.xticks(x_ticks_positions, x_ticks_labels)
    plt.savefig(pic_name)


#dd = np.array([[1.1,2.1,3.1],[3.1,4.5,5.5]])
#print(dd.shape)
#print_hist(dd)