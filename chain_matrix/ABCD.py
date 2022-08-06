import math
import numpy as np
import matplotlib.pyplot as plt

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

f = 100
omiga = 2 * math.pi * f

A_value = chainMatrix.A(omiga)
B_value = chainMatrix.B(omiga)
C_value = chainMatrix.C(omiga)
D_value = chainMatrix.D(omiga)

print(A_value,abs(A_value))
print(B_value,abs(B_value))
print(C_value,abs(C_value))
print(D_value,abs(D_value))

matrix_names = ["A","B","C","D"]

def generate_element_chain_matrix(matrix_name):
    sample_count = 256
    chainMatrix = ChainMatrix()
    matrix_func = chainMatrix.__getattribute__(matrix_name)
    frequency_points = list(map(lambda x:np.abs(matrix_func(2*math.pi*x)),range(1,sample_count)))
    frequency_points.insert(0,0) # the magnitude at 0 frequency is value
    print(frequency_points)
    ind = np.arange(sample_count)
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)
    ax1.bar(ind, frequency_points, width, label='Magnitude')
    #for i, (x, y) in enumerate(zip(ind, frequency_points)):
    #    ax1.text(x, y, '%.2f' % y, ha='center', va='bottom')


    ax1.set_ylabel('magnitude')
    ax1.set_xlabel('frequency')
    if matrix_name == "A" or matrix_name == "D":
        ax1.set_ylim(ymin = 1-6e-4,ymax = 1+6e-4)
    plt.title('%s in the chain matrix'%matrix_name)

    #plt.xticks(ind + width / 2, ('train_accuracy', 'val_accuracy', 'test_accuracy', 'training_time(seconds)'))
    plt.legend(loc='best')
    plt.savefig("%s.png"%matrix_name,dpi=300)


for matrix_name in matrix_names:
    generate_element_chain_matrix(matrix_name)


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
