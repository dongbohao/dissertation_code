import numpy
from scipy import signal
import matplotlib.pyplot as plt
import math
import numpy as np
import librosa
import torch

F_0 = 130
a_m = 2/3
t_p = 0.8*8 /1000 * a_m   # s
t_n = 0.8*8 /1000 - t_p    # s
t_0 = 8 /1000  # s
A = 4/1000
O_q = 0.8


print("TTT one period",t_p,t_n,t_0)
def rosenberg_c(t):
    if 0<= t <= t_p:
        ret = A/2*(1-math.cos(math.pi*t/t_p))
    elif t_p< t <= t_p + t_n:
        ret = A*math.cos(math.pi/2 * (t-t_p)/t_n )
    elif t_p + t_n < t <= t_0:
        ret = 0
    return ret


def one_period():
    samples = np.linspace(0,8/1000,int(8/1000 * 22500))
    t = np.linspace(1,int(8/1000 * 22500),int(8/1000 * 22500))
    ll = list(map(lambda sample:rosenberg_c(sample),samples))
    return ll
#print(ll[:10])


def get_fft():
    ll = one_period()
    ll = np.array(ll)
    fft = librosa.stft(ll,n_fft=1024)
    return abs(fft[1:])

fft_r = get_fft()
fft_r = torch.tensor(fft_r).squeeze().unsqueeze(0).unsqueeze(0)

def get_torch_fft(batch_size,mel_length,n_fft):
    target = torch.ones(batch_size,mel_length,n_fft)
    target = target*fft_r
    return target




def one_second():
    n = 1
    ll = []
    for i in range(n):
        ll += one_period()

    ll = one_period()
    t = np.linspace(0,len(ll),len(ll))

    plt.plot(t, ll,label="Rosenberg_c")
    plt.legend( loc='best')
    plt.xlabel("Samples(time) by 22500 sample rate")
    plt.ylabel("Amplitude")
    plt.grid(True)
    pic_name = "rosenberg_c_1_period"
    plt.savefig("v9_%s.png"%pic_name)


def draw_fft():
    fft = get_fft()
    t = np.linspace(1,len(fft),len(fft))
    plt.plot(t, fft, label="Rosenberg_c")
    plt.legend(loc='best')
    plt.xlabel("Frequence")
    plt.ylabel("Magnitude")
    plt.grid(True)
    pic_name = "rosenberg_c_fft"
    plt.savefig("v9_%s.png" % pic_name)


#one_second()
#draw_fft()
#get_torch_fft(2,33,512)
#draw_fft()


def rosenberg_c_paramter(t,A,t_p,t_n,t_0):
    if 0<= t <= t_p:
        ret = A/2*(1-math.cos(math.pi*t/t_p))
    elif t_p< t <= t_p + t_n:
        ret = A*math.cos(math.pi/2 * (t-t_p)/t_n )
    elif t_p + t_n < t <= t_0:
        ret = 0
    return ret


def get_rosenberg_waveform(duration=0,is_voiced=True,sr = 22500,t_0=1/125,A=4/1000,t_p=0.8*1/125 * 2/3,t_n=0.8*1/125 * 1/3,O_q=0.8,a_m=2/3,F_0=125):
    """
    duration: second
    sr: sample rate, 22500
    """
    period_count = math.ceil(duration/t_0)
    ll = []
    #return ll
    rest_duration = duration
    for p in range(period_count):
        duration_t = min(rest_duration,t_0)
        samples = np.linspace(0,duration_t,math.ceil(duration_t * sr))
        if is_voiced:
            ll += list(map(lambda t:rosenberg_c_paramter(t,A,t_p,t_n,t_0),samples))
        else:
            ll += list(map(lambda t: 0, samples))
        rest_duration -= t_0
    return ll



