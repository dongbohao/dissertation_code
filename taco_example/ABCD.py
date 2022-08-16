import math

import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch

from tacotron2_common.utils import to_gpu

class ChainMatrix():

    def __init__(self,c_tract = 4, areat_tract = 1.543):
        self.c = 34300
        self.delta_l = 0.85
        self.a = 130 * math.pi
        self.b = math.pow(30 * math.pi, 2)
        self.omiga_zero = math.pow(406 * math.pi, 2)
        self.c_tract = c_tract
        #self.c_nasal = 72
        self.rho = 1.86e-5
        self.area_tract = areat_tract


    def Upsilon(self,omiga):
        return np.power(1j*omiga*self.c_tract,0.5)

    def Psi(self,omiga):
        part_1 = 1j*omiga*self.omiga_zero/((1j*omiga+self.a)*1j*omiga + self.b)
        part_2 = self.Upsilon(omiga)
        return part_1 + part_2

    def Theta(self,omiga):
        return np.power((self.a+1j*omiga)/(self.Psi(omiga) + 1j*omiga),0.5)

    def Phi(self,omiga):
        return self.Theta(omiga)*(self.Psi(omiga) + 1j*omiga)

    def A(self,omiga):
        return np.cosh(self.Phi(omiga)*self.delta_l/self.c)

    def B(self,omiga):
        return -self.rho*self.c/self.area_tract*self.Theta(omiga)*np.sinh(self.Phi(omiga)*self.delta_l/self.c)

    def C(self,omiga):
        return -self.area_tract/(self.rho*self.c)*np.sinh(self.Phi(omiga)*self.delta_l/self.c)/self.Theta(omiga)

    def D(self,omiga):
        return np.cosh(self.Phi(omiga)*self.delta_l/self.c)



chainMatrix = ChainMatrix()

# f = 100
# omiga = 2 * math.pi * f
#
# A_value = chainMatrix.A(omiga)
# B_value = chainMatrix.B(omiga)
# C_value = chainMatrix.C(omiga)
# D_value = chainMatrix.D(omiga)
#
# print(A_value,abs(A_value))
# print(B_value,abs(B_value))
# print(C_value,abs(C_value))
# print(D_value,abs(D_value))

matrix_names = ["A","B","C","D"]


def print_fre(stft_list,n_fft=1024,sr=22500,pic_name=""):


    ind = list(map(lambda x:x, np.arange(int(n_fft/2))))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

    ax1.bar(ind, stft_list, width, label='Magnitude')
    #for i, (x, y) in enumerate(zip(ind, frequency_points)):
    #    ax1.text(x, y, '%.2f' % y, ha='center', va='bottom')


    ax1.set_ylabel('magnitude')
    ax1.set_xlabel('frequency')

    x_ticks_positions = [n for n in range(0, n_fft // 2, n_fft // 16)]
    x_ticks_labels = [str(sr / 512 * n) + 'Hz' for n in x_ticks_positions]
    plt.xticks(x_ticks_positions,x_ticks_labels)
    # if matrix_name == "A" or matrix_name == "D":
    #     ax1.set_ylim(ymin = 1-6e-4,ymax = 1+6e-4)
    plt.title('')

    #plt.xticks(ind + width / 2, ('train_accuracy', 'val_accuracy', 'test_accuracy', 'training_time(seconds)'))
    plt.legend(loc='best')
    #plt.savefig("%s.png"%,dpi=300)
    plt.savefig("%s.png"%pic_name,dpi=300)


def generate_element_chain_matrix(matrix_name,sample_rate= 22050, n_fft=1024, mel_channels = 80):
    chainMatrix = ChainMatrix()
    matrix_func = chainMatrix.__getattribute__(matrix_name)
    frequency_points = list(map(lambda x:np.abs(matrix_func(2*math.pi*x)),filter(lambda x:x>0,np.linspace(0,sample_rate+1,int(n_fft/2)+1, dtype=int) )))
    frequency_points.insert(0,0) # the magnitude at 0 frequency is value
    # print(frequency_points)
    # ind = np.arange(int(n_fft/2+1))
    # width = 0.35
    # fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)
    # ax1.bar(ind, frequency_points, width, label='Magnitude')
    # #for i, (x, y) in enumerate(zip(ind, frequency_points)):
    # #    ax1.text(x, y, '%.2f' % y, ha='center', va='bottom')
    #
    #
    # ax1.set_ylabel('magnitude')
    # ax1.set_xlabel('frequency')
    # if matrix_name == "A" or matrix_name == "D":
    #     ax1.set_ylim(ymin = 1-6e-4,ymax = 1+6e-4)
    # plt.title('%s in the chain matrix'%matrix_name)
    #
    # #plt.xticks(ind + width / 2, ('train_accuracy', 'val_accuracy', 'test_accuracy', 'training_time(seconds)'))
    # plt.legend(loc='best')
    # plt.savefig("%s.png"%matrix_name,dpi=300)

    print_fre(frequency_points[:-1],pic_name=matrix_name)
    m = torch.tensor([[frequency_points]]).float()
    m = to_gpu(m).float()
    print("Chain matrix",m.size())
    return m


def make_chain_matrix(sample_rate= 22050, n_fft=1024, mel_channels = 80):
    ret = {}
    for matrix_name in matrix_names:
        m = generate_element_chain_matrix(matrix_name, sample_rate=sample_rate,n_fft=n_fft,mel_channels=mel_channels)
        ret[matrix_name] = m
    return ret


# aa = make_chain_matrix()
#
# print(aa["A"].size())
#
#
# import torch
# tensor1 = torch.randn(2,2,3)
# print(tensor1[0,0])
#
# tensor3 = torch.randn(3)
#
# tensor2 = torch.randn(8,3)
# print(tensor1[0,0])
#
# rt = torch.matmul(tensor2,tensor3)
# print(rt.size())
#
#
#
#
# tensor5 = torch.randn(2,2,438,513)
# tensor6 = torch.randn(2,438,513)
#
#
# tensor8 = torch.tensor([[[1,2,3]]])
# print(tensor8.size())
# aa = tensor8.repeat(1,9,1)
# print(aa.size())
#
#
#
# tensor9 = torch.tensor([[[[1,2,3]]]])
# tensor10 = torch.tensor([[[[1,2,3],[1,2,3],[1,2,3]]]])
#
# print(tensor9.size())
# print(tensor10.size())
#
#
# rrrt = tensor10*tensor9
# print(rrrt)



# tt1 = torch.randn(2,2,5)
# tt2 = torch.randn(2,2,5)
# rr = torch.matmul(tt1,tt2)
# print(rr.size())

#for matrix_name in matrix_names:
#     generate_element_chain_matrix(matrix_name)


# chainMatrix = ChainMatrix()
# frequency_points = list(map(lambda x:np.abs(chainMatrix.A(2*math.pi*x)),range(1,257)))
#
#
# time_domian = np.fft.ifft(frequency_points)
# print(time_domian)
#
# t = np.arange(256)
# plt.plot(t, time_domian.real, label='aa')
# plt.show()




import librosa
# load the file

#print_fre([])