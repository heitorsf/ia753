import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from scipy.signal import buttord,butter,freqz,lfilter,iirnotch

# Filtros
PB_filt = True
PA_filt = True
Notch_filt = True

# Plots
plotQ1 = True
plotQ2a = True
plotQ2b = False
plotQ2c = True
plotQ2d = True

# 
mysignal = np.loadtxt('../data/signal.txt')
fs = 500  # Hz
dt = 1./fs # seconds
t = np.linspace(0,dt*mysignal.shape[0],mysignal.shape[0],endpoint=False)
N = len(mysignal)


# Leitura do sinal original
sig_fft = np.fft.rfft(mysignal)
f_dimensionless = np.fft.rfftfreq(len(mysignal))
T = dt*N
df = 1./T
fhz_nf = f_dimensionless*df*N

# Questão 2
# (a)
if PB_filt:
    lp_fpass = 80.
    lp_fstop = 120.
    fny = fs/2.

    lp_ord, lp_wn = buttord(lp_fpass/fny,lp_fstop/fny,3,40)
    lp_b, lp_a = butter(lp_ord, lp_wn, 'lowpass')
    lp_w, lp_h = freqz(lp_b, lp_a, worN=len(fhz_nf),fs=fs*2*np.pi)
    
    sig_lp = lfilter(lp_b,lp_a,mysignal)
    
    sig_fft_lp = np.fft.rfft(sig_lp)
    f_dimensionless_lp = np.fft.rfftfreq(len(sig_lp))
    T = dt*N
    df = 1./T
    fhz_lp = f_dimensionless_lp*df*N
    
    if plotQ2a:
        fig5 = plt.figure(figsize=(8,4))
        plt.plot(lp_w/(2*np.pi), 20 * np.log10(abs(lp_h)))
        plt.xscale('log')
        plt.title('Filtro Butterworth PB')
        plt.xlabel(u'Frequência [Hz]')
        plt.ylabel(u'Amplitude [dB]')
        plt.ylim((-200,50))
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(lp_fpass, color='green') # cutoff frequency

# (b) Passa-altas
if PA_filt:
    afiltrar_PA = sig_lp
    hp_fpass = .5
    hp_fstop = .1
    fny = fs/2.
    
    hp_ord, hp_wn = buttord(hp_fpass/fny,hp_fstop/fny,3,40)
    
    hp_b, hp_a = butter(4, hp_fpass/fny, 'highpass')
    hp_w, hp_h = freqz(hp_b, hp_a, worN=len(fhz_lp),fs=fs*2*np.pi)
    
    sig_lphp= lfilter(hp_b,hp_a,sig_lp)
    
    sig_fft_lphp = np.fft.rfft(sig_lphp)
    f_dimensionless_lphp = np.fft.rfftfreq(len(sig_lphp))
    T = dt*N
    df = 1./T
    fhz_lphp = f_dimensionless_lphp*df*N
    
# Questão 2 (c)
if Notch_filt:
    notch_f = 50.3
    fny = fs/2.
    
    notch_b, notch_a = iirnotch(notch_f/fny, 10)
    notch_w, notch_h = freqz(notch_b, notch_a, worN=len(fhz_nf),fs=fs*2*np.pi)
    
    sig_notch = lfilter(notch_b,notch_a,sig_lphp)
    
    sig_fft_notch = np.fft.rfft(sig_notch)
    f_dimensionless_notch = np.fft.rfftfreq(len(sig_notch))
    T = dt*N
    df = 1./T
    fhz_notch = f_dimensionless_notch*df*N
    
    fig12 = plt.figure()
    plt.subplot(2,1,1)
    plt.xlabel('Tempo [s]')
    plt.ylabel('Sinal ECG [V]')
    plt.plot(t,mysignal,label='Original')
    plt.legend()
    #
    plt.subplot(2,1,2)
    plt.xlabel('Tempo [s]')
    plt.ylabel('Sinal ECG [V]')
    plt.plot(t,sig_notch,label='Filtrado (combinado)')
    plt.legend()
    #

    fig10 = plt.figure(figsize=(14,4.5))
    plt.title('Q2 (c)')
    plt.subplot(2,1,1)
    plt.xlabel(u'frequência [Hz]')
    plt.ylabel('FFT sinal original ')
    plt.plot(fhz_nf,sig_fft,label='Original')
    plt.tight_layout()
    plt.legend()
    #
    plt.subplot(2,1,2)
    plt.xlabel(u'frequência [Hz]')
    plt.ylabel('FFT sinal filtrado')
    plt.plot(fhz_notch,sig_fft_notch,label='Filtrado (combinado)')
    plt.legend()
    plt.tight_layout()
