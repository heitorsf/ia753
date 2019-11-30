import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from scipy.signal import buttord,butter,freqz,lfilter,iirnotch

mysignal = np.loadtxt('../data/signal_filtered.txt')
fs = 500  # Hz
dt = 1./fs # seconds
t = np.linspace(0,dt*mysignal.shape[0],mysignal.shape[0],endpoint=False)
N = len(mysignal)


# Questão 3
sig_fft = np.fft.rfft(mysignal)
f_dimensionless = np.fft.rfftfreq(len(mysignal))
T = dt*N
df = 1./T
fhz_nf = f_dimensionless*df*N

argpeaks = []
for i in range(1,len(mysignal)-1):
    if mysignal[i] > 0.3:
        if mysignal[i] > mysignal[i-1]:
            if mysignal[i] > mysignal[i+1]:
                argpeaks.append(i)
peaks_t = t[argpeaks]
peaks_v = mysignal[argpeaks]

impulses = np.array([1./dt if instant in peaks_t else 0 for instant in t])

w =  1200.
window = np.ones(int(w))/w

ifreq = np.convolve(window,impulses)

fig1 = plt.figure(figsize=(14,4.5))
plt.title('Q3 (a)')
plt.xlabel('Tempo [s]')
plt.ylabel('Sinal ECG filtrado [V]')
plt.plot(t,mysignal)
plt.plot(peaks_t,peaks_v,'r^',markersize=5)
plt.grid(b=True,which='both')
plt.tight_layout()
plt.savefig('../images/Q3a.png')

fig2 = plt.figure(figsize=(8,5))
plt.title('Q3 (b)')
plt.xlabel('Tempo [s]')
plt.ylabel('Trem de impulsos')
plt.ylim((-5,550))
plt.plot(t,impulses,'.')
plt.tight_layout()
plt.savefig('../images/Q3b.png')


fig3 = plt.figure(figsize=(8,5))
plt.title('Q3 (c)')
plt.xlabel('Tempo [s]')
plt.ylabel(u'Frequência instantânea [Hz]')
#plt.ylim((-5,550))
plt.plot(ifreq,label='w = '+str(int(w)))
plt.legend()
plt.tight_layout()
plt.savefig('../images/Q3c.png')
