import numpy
from scipy import signal
import matplotlib.pyplot as plt
import math
import numpy as np

alpha_2 = math.exp(-0.069/(0.00017*16000))
f0 = 1/(2*math.pi)*math.pow(34/0.00017,0.5)  # Frequency to be retained (Hz)
fs = 16000  # Sample frequency (Hz)
Q = -f0*math.pi/(fs*math.log(math.pow(alpha_2,0.5)))  # Quality factor
# Design peak filter
print(f0,Q,fs)
#f0=80
#Q=3.3
#f0=71
b, a = signal.iirpeak(f0, Q, fs)



n = 200

xn = np.ones(300)*2
# stp = np.linspace(0.3,2,150)
# for i in range(0,150):
#     xn[i] = stp[i]
xn = xn


#print(xn.shape)
#print(xn)

def fc(xn):
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
    #z2, _ = signal.lfilter(b, a, z, zi=zi * z[0])
    return z



x_01 = 0.01
x_02 = 0.009


def x1(x_n):
    x_n_ = x_n.copy()
    x_n_[x_n>-x_01] += 0
    x_n_[x_n<=-x_01] = - x_01
    return x_n_

def x2(x_n):
    x_1 = x1(x_n)
    last_x_1 = np.ones(1)*x_01
    x_1_d = np.diff(np.hstack([last_x_1, x_1]))*16000
    x_2 = x_1 + T*x_1_d/c_f
    x_2_ = x_2.copy()
    x_2_[x_2>-x_02] += 0
    x_2_[x_2<=-x_02] = - x_02

    #print(x_1)
    #print(x_2_)
    return x_2_



L = 0.014
T = 1.3
c_f = 1000
P_l = 2
p = 1.15
gama = 1.3

def a1(x_1):
    ret = 2 * L * (x_01 + x_1)
    return ret

def a2(x_2):
    ret = 2 * L * (x_02 + x_2)
    return ret


def Ug(a_1,a_2):
    coef = gama*a_1*(gama*a_1<a_2) + a_2*(a_2<gama*a_1)
    print("CCC",coef.shape)
    ret = math.pow(2*P_l/p,0.5) * coef
    return ret

# def Pm(U_g,a_1):
#     Pl = np.ones(300)*2
#     ret = Pl - 0.5*p * np.power(U_g,2)/np.power(a_1,2)
#     return ret


def Pm(a_1,a_2):
    pm = np.zeros(300)
    pm[a_1==0] = P_l
    idx1 = (gama*a_1>=a_2) * (a_2>=0)
    print(sum(idx1))
    pm[idx1] = P_l*(1-np.power(a_2[idx1],2)/np.power(a_1[idx1],2))
    idx2 = (gama*a_1 > 0) * (a_2> gama * a_1)
    print(sum(idx2))
    pm[idx2] = P_l * (1 - np.power(a_2[idx2], 2) / np.power(a_1[idx2], 2))

    return pm



ll = []
ul = []
xl = []
xll = []
xnl = []
p_m = xn
xnn = []

last_x1 = np.zeros(1)
for abc in range(0,5):
    x_n = fc(xn)
    #print("XXXXXn",x_n)
    xnl.append(x_n)
    x_1 = x1(x_n)
    x_2 = x2(x_n)
    a_1 = a1(x_1)
    a_2 = a2(x_2)
    u_g = Ug(a_1,a_2)
    last_x1 = x_1[x_1.shape[0]-1:]
    #p_m = Pm(u_g,a_1)
    p_m = Pm(a_1,a_2)
    ll.append(p_m)
    ul.append(u_g)
    xl.append(x_1)
    xll.append(x_2)
    xnn.append(x_n)
    #print("11111111",x_1)
    #print("22222222",x_2)

    print("Loop",p_m.shape)


#t = np.linspace(0, 1*300*2/16000, 300*2)
t = np.arange(0,300*5)
z = numpy.hstack(ll)
z_g = numpy.hstack(ul)
z_x1 = numpy.hstack(xl)
z_x2 = numpy.hstack(xll)
z_xn = numpy.hstack(xnn)

print(z_x1[:10])
print(z_x2[:10])

z_n = numpy.hstack(xnl)
print(z.shape)


plt.figure
#plt.plot(t, xn, 'b', alpha=0.75)
plt.plot(t, z, 'r--')
#plt.plot(t, z_g, 'r--')
#plt.plot(t, z_x1, 'r--')
#plt.plot(t, z_x2)
#plt.plot(t, z_xn)
#plt.plot(t, z_n, 'r--')
plt.legend(('noisy signal', 'IIR, once'), loc='best')
plt.grid(True)
plt.show()