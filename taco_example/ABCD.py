import math

import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
from functools import reduce

from tacotron2_common.utils import to_gpu

c = 34300 # cm/s
delta_l = 0.85 # cm/s


class ChainMatrix():

    def __init__(self,c_tract = 4, areat_tract = 1.543):
        self.c = c
        self.delta_l = delta_l
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

matrix_names = ["A","B","C","D"]


def print_fre(stft_list,n_fft=1024,sr=22500,pic_name=""):


    ind = list(map(lambda x:x, np.arange(int(n_fft/2))))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

    ax1.plot(ind, stft_list, label='Magnitude')
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
    #plt.show()

def generate_element_chain_matrix(matrix_name,sample_rate= 22050, n_fft=1024, mel_channels = 80):
    chainMatrix = ChainMatrix()
    matrix_func = chainMatrix.__getattribute__(matrix_name)
    frequency_points = list(map(lambda x:np.abs(matrix_func(2*math.pi*x)),filter(lambda x:x>0,np.linspace(0,sample_rate+1,int(n_fft/2)+1, dtype=int) )))
    frequency_points.insert(0,0) # the magnitude at 0 frequency is zero
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



def broadcast_inner_product(matrix_left, matrix_right):
    A = matrix_left[0,0] * matrix_right[0,0] + matrix_left[1,0] * matrix_right[0,1]
    B = matrix_left[0,1] * matrix_right[0,0] + matrix_left[1,1] * matrix_right[0,1]
    C = matrix_left[0,0] * matrix_right[1,0] + matrix_left[1,0] * matrix_right[1,1]
    D = matrix_left[0,1] * matrix_right[1,0] + matrix_left[1,1] * matrix_right[1,1]

    #print(A.tolist())

    ret = torch.tensor([[A.tolist(),B.tolist()],[C.tolist(),D.tolist()]])
    return ret


L_G = 4.32    # cm
L_L = 3.83    # cm
L_N = 9.26    # cm
L_C = 5.43    # cm

Area_G = 2.04  # cm^2
Area_L = 3.52  # cm^2
Area_N = 2.5   # cm^2
Area_C = 1.54  # cm^2
c_tract_G = 4
c_tract_L = 4
c_tract_N = 72
c_tract_C = 4


def get_section_matrix(tract_name,sample_rate = 22500,n_fft=1024):
    if tract_name == "G":
        area = Area_G
        c_tract = c_tract_G

    if tract_name == "L":
        area = Area_L
        c_tract = c_tract_L

    if tract_name == "N":
        area = Area_N
        c_tract = c_tract_N

    if tract_name == "C":
        area = Area_C
        c_tract = c_tract_C

    chainMatrix = ChainMatrix(c_tract = c_tract, areat_tract = area)
    matrix_list = []
    for matrix_name in matrix_names:
        matrix_func = matrix_func = chainMatrix.__getattribute__(matrix_name)
        frequency_points = list(map(lambda x: matrix_func(2 * math.pi * x), filter(lambda x: x > 0,np.linspace(0, sample_rate + 1,int(n_fft / 2) + 1,dtype=int))))
        matrix_list.append(frequency_points)
        # frequency_points.insert(0, 0)

    K = torch.tensor([[matrix_list[0],matrix_list[1]],[matrix_list[2],matrix_list[3]]])
    print("K",tract_name,K.size())
    return K


def get_tract_matrix(tract_name):
    if tract_name == "G":
        section_count = math.ceil(L_G/delta_l)
        K = get_section_matrix(tract_name)
        K_tract = reduce(lambda matrix_left,matrix_right:broadcast_inner_product(matrix_left,matrix_right),[K]*section_count)

    elif tract_name == "L":
        section_count = math.ceil(L_L / delta_l)
        K = get_section_matrix(tract_name)
        K_tract = reduce(lambda matrix_left, matrix_right: broadcast_inner_product(matrix_left, matrix_right),
                         [K] * section_count)
    elif tract_name == "N":
        section_count = math.ceil(L_N / delta_l)
        K = get_section_matrix(tract_name)
        K_tract = reduce(lambda matrix_left, matrix_right: broadcast_inner_product(matrix_left, matrix_right),
                         [K] * section_count)
    elif tract_name == "C":
        section_count = math.ceil(L_C / delta_l)
        K = get_section_matrix(tract_name)
        K_tract = reduce(lambda matrix_left, matrix_right: broadcast_inner_product(matrix_left, matrix_right),
                         [K] * section_count)

    return K_tract


k_vg = 0.2 # cm
k_vt = 0.2 # cm
k_vn = 0.2 # cm

def impedence(velum_name,omiga):
    if velum_name == "kvn":
        z_vn = pow(k_vn*omiga/c,2)/2 + 1j*8*(k_vn*omiga/c)/(3*math.pi)
        return z_vn
    if velum_name == "kvt":
        z_vt = pow(k_vt*omiga/c,2)/2 + 1j*8*(k_vt*omiga/c)/(3*math.pi)
        return z_vt
    if velum_name == "kvg":
        z_l = pow(k_vg * omiga / c, 2) / 2 + 1j * 8 * (k_vg * omiga / c) / (3 * math.pi)
        return z_l


def impendence_frequency(velum_name,K_tract_sections,sample_rate=22500,n_fft=1024):
    A = K_tract_sections[0, 0].squeeze()
    B = K_tract_sections[0, 1].squeeze()
    C = K_tract_sections[1, 0].squeeze()
    D = K_tract_sections[1, 1].squeeze()
    if velum_name == "kvg":
        Z_g = list(map(lambda x: impedence(velum_name,2 * math.pi * x), filter(lambda x: x > 0,np.linspace(0, sample_rate + 1,int(n_fft / 2) + 1,dtype=int))))
        Z_g = torch.tensor(Z_g)
        Z_in_g = (D*Z_g-B)/(A - C*Z_g)
        return Z_in_g

    if velum_name == "kvn":
        Z_vn = list(map(lambda x: impedence(velum_name,2 * math.pi * x), filter(lambda x: x > 0,np.linspace(0, sample_rate + 1,int(n_fft / 2) + 1,dtype=int))))
        Z_vn = torch.tensor(Z_vn)
        Z_in_vn = (D*Z_vn-B)/(A - C*Z_vn)
        return Z_in_vn

    if velum_name == "kvt":
        Z_vt  = list(map(lambda x: impedence(velum_name,2 * math.pi * x), filter(lambda x: x > 0,np.linspace(0, sample_rate + 1,int(n_fft / 2) + 1,dtype=int))))
        Z_vt = torch.tensor(Z_vt)
        Z_in_vt = (D*Z_vt-B)/(A - C*Z_vt)
        return Z_in_vt

def get_velum_matrix(velum_name,K_tract_sections,sample_rate=22500,n_fft=1024):
        if velum_name == "kvn":
            Z_in_vn = impendence_frequency(velum_name,K_tract_sections,sample_rate=sample_rate,n_fft=n_fft)
            K_cn = torch.tensor([[[1] * 512, [0] * 512], [-1/Z_in_vn, [1] * 512]])
            return K_cn
        if velum_name == "kvt":
            Z_in_vt = impendence_frequency(velum_name,K_tract_sections,sample_rate=sample_rate,n_fft=n_fft)
            K_ct = torch.tensor([[[1] * 512, [0] * 512], [-1/Z_in_vt, [1] * 512]])
            return K_ct
        if velum_name == "kvg":
            Z_in_g = impendence_frequency(velum_name,K_tract_sections,sample_rate=sample_rate,n_fft=n_fft)
            K_vg = torch.tensor([[[1] * 512, [0] * 512], [-1/Z_in_g, [1] * 512]])
            return K_vg

area1 = 0.15 # cm^2
a_vib = 5 # cm
def get_vibration_transfer_function(K_G,sample_rate=22500,n_fft=1024):
    chainMatrix = ChainMatrix(c_tract=c_tract_G,areat_tract=Area_G)
    Z_g = impendence_frequency("kvg",K_G)
    beta = chainMatrix.Psi
    #z_coef = area1/c * 1j*omiga*a_vib/(c+1j*omiga*a_vib) * beta(omiga)
    z_coef = list(map(lambda x: area1/c * 1j* 2* math.pi * x *a_vib/(c+1j* 2 * math.pi * x *a_vib) * beta(2*math.pi*x),
               filter(lambda x: x > 0, np.linspace(0, sample_rate + 1, int(n_fft / 2) + 1, dtype=int))))
    H_vib = torch.tensor(z_coef) * Z_g
    return H_vib





K_G= get_tract_matrix("G")
#print("K_G",K_G.size())
K_L= get_tract_matrix("L")
#print("K_L",K_G.size())
K_C= get_tract_matrix("C")
#print("K_C",K_G.size())
K_N= get_tract_matrix("N")
#print("K_N",K_G.size())

K_cn = get_velum_matrix("kvn",K_G)
#print("K_cn",K_cn.size())
K_ct = get_velum_matrix("kvt",K_G)
#print("K_ct",K_ct.size())


K_tract = reduce(lambda matrix_left,matrix_right:broadcast_inner_product(matrix_left,matrix_right),[K_L,K_C,K_cn,K_G])
K_nasal = reduce(lambda matrix_left,matrix_right:broadcast_inner_product(matrix_left,matrix_right),[K_N,K_ct,K_G])

H_vib = get_vibration_transfer_function(K_G)
# print("K_tract",K_tract.size())
# print("K_nasal",K_nasal.size())
# print("H_vib",H_vib.size())

# print_fre(torch.abs(H_vib).squeeze().tolist(),pic_name="H_vib")
#
# print_fre(torch.abs(K_tract[0,0]).squeeze().tolist(),pic_name="vocal_tract_chainMatrix_A")
# print_fre(torch.abs(K_tract[0,1]).squeeze().tolist(),pic_name="vocal_tract_chainMatrix_B")
# print_fre(torch.abs(K_tract[1,0]).squeeze().tolist(),pic_name="vocal_tract_chainMatrix_C")
# print_fre(torch.abs(K_tract[1,1]).squeeze().tolist(),pic_name="vocal_tract_chainMatrix_D")
#
#
# print_fre(torch.abs(K_nasal[0,0]).squeeze().tolist(),pic_name="nasal_tract_chainMatrix_A")
# print_fre(torch.abs(K_nasal[0,1]).squeeze().tolist(),pic_name="nasal_tract_chainMatrix_B")
# print_fre(torch.abs(K_nasal[1,0]).squeeze().tolist(),pic_name="nasal_tract_chainMatrix_C")
# print_fre(torch.abs(K_nasal[1,1]).squeeze().tolist(),pic_name="nasal_tract_chainMatrix_D")

#matrix_left = torch.tensor([[1,2],[3,4]])
#matrix_right = torch.tensor(([5,6],[7,8]))
#matrix_left = torch.randn(2,2,12)
#matrix_right = torch.randn(2,2,12)

#aa = torch.matmul(matrix_left,matrix_right)
#print(aa)
#bb = broadcast_inner_product(matrix_left,matrix_right)

#print(bb)
#print(bb.size())
