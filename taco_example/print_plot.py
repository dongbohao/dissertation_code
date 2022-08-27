import librosa
import librosa.display
import matplotlib.pyplot as plt

version = "v9_vv"


def print_spectrogram(pred_y_mel,pic_name = "",sample_rate=22500,hop_length=256):
    pred_y_mel = pred_y_mel.to("cpu").squeeze().detach().numpy().T
    log_spectro = pred_y_mel
    print("mel",pred_y_mel.shape)
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(log_spectro, sr=sample_rate, x_axis='time', y_axis='hz', hop_length=hop_length,
                             cmap='magma', fmax=80)
    plt.colorbar(label='Decibels')
    if pic_name:
        plt.savefig("%s_%s.png"%(version,pic_name))


