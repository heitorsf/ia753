# -*- coding: utf-8 -*-
"""
Heitor Sanchez Fernandes - IA753
# Trabalho Computacional 2

## Segunda Tarefa

"""

import numpy as np; np.set_printoptions(precision=3)
import matplotlib.pyplot as plt;plt.ion()
from matplotlib.ticker import StrMethodFormatter
from scipy.io import loadmat
from scipy.signal import csd
#import sys
plt.rc('font',size=12)

#myneuron = int(sys.argv[1])

print('\nSegunda tarefa')

# (a)
print('\n(a)')

# Carregar o arquivo
data = loadmat('../data/tc2ex2.mat')

print("Elementos do arquivo tc2ex2.mat: ")
print([key for key in data.keys()])

Fs = data['Fs'][0][0]
x = np.array(data['input'])
y = np.array(data['output'])

fxy,Sxy = csd(x.transpose(),y.transpose(),fs=Fs,scaling='spectrum',nperseg=100000)
fxx,Sxx = csd(x.transpose(),x.transpose(),fs=Fs,scaling='spectrum',nperseg=100000)
H = Sxy[0]/Sxx[0]
if all(fxy==fxx):
    f = fxx
else:
    raise ValueError(u'As frequencias amostradas de Sxy e Sxx nao sao correspondentes, verificar.')

fig1,axs = plt.subplots(2,1)
#plt.xlabel(u'Frequências amostradas [Hz]')
axs[0].semilogx(f,20*np.log10(np.abs(H)))
axs[0].set_ylabel(r'$|\hat{H}(jw)|$ [dB]') # Resposta em frequência estimada')
axs[0].set_xlim(xmax=100)
axs[0].set_ylim(ymin=-75)
axs[0].grid(which='both')

#axs[1].semilogx(f,np.angle(H,deg=True),'.-')
axs[1].semilogx(f,np.rad2deg(np.unwrap(np.angle(H))))
axs[1].set_xlabel(u'Frequências amostradas [Hz]')
axs[1].set_ylabel(r'$\angle{\hat{H}}(jw)$') # Resposta em frequência estimada')
axs[1].set_xlim(xmax=100)
axs[1].set_ylim((-220,10))
axs[1].grid(which='both')
axs[1].yaxis.set_major_formatter(StrMethodFormatter(r'{x:.0f}$\degree$'))

plt.tight_layout()
plt.savefig('../images/bode_itemb.png')

# (b)
print('\n(b)')

wn_ang = np.rad2deg(np.unwrap(np.angle(H)))
wn_idx = np.argwhere(wn_ang<=-90)[0]
plt.vlines(2,-300,100,'green')
#print(u'    Média: %.3f N'%force_avg)

# (c) Estimativa de Frequência instantânea
print('\n(c)')

wn = 2 #Hz
w = 2*np.pi*f
Hwn = 4./(-np.power(w,2) + (0+4j)*w + 4)

fig2,axs2 = plt.subplots(2,1)
#plt.xlabel(u'Frequências amostradas [Hz]')
axs2[0].semilogx(f,20*np.log10(np.abs(Hwn)),label=r'Calculado com \omega_n')
axs2[0].semilogx(f,20*np.log10(np.abs(H)),label='Estimado')
axs2[0].set_ylabel(r'$|H(jw)|$ [dB]') # Resposta em frequência estimada')
axs2[0].set_xlim(xmax=100)
axs2[0].set_ylim(ymin=-75)
axs2[0].grid(which='both')

#axs2[1].semilogx(f,np.angle(H,deg=True),'.-')
axs2[1].semilogx(f,np.rad2deg(np.unwrap(np.angle(Hwn))),label=r'Calculado com \omega_n')
axs2[1].semilogx(f,np.rad2deg(np.unwrap(np.angle(H))),label='Estimado')
axs2[1].set_xlabel(u'Frequências amostradas [Hz]')
axs2[1].set_ylabel(r'$\angle{H}(jw)$') # Resposta em frequência estimada')
axs2[1].set_xlim(xmax=100)
axs2[1].set_ylim((-220,10))
axs2[1].grid(which='both')
axs2[1].yaxis.set_major_formatter(StrMethodFormatter(r'{x:.0f}$\degree$'))


plt.tight_layout()
plt.savefig('../images/bodec.png')

#print('%d & %.2f & %.2f & %.2f & %.2f & %.2f \\\\'%(neurons_id[i],isis_avg[i],isis_std[i],isis_cv[i],isis_skew[i],isis_kurt[i]))
