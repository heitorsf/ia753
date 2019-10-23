import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import scipy.signal as sgn

q1 = True
q2 = True

signal = np.loadtxt('signal.txt')
fs = 500  # Hz
dt = 1./fs # seconds
t = np.linspace(0,dt*signal.shape[0],signal.shape[0],endpoint=False)
N = len(signal)


# Questão 1
sig_fft = np.fft.rfft(signal)
f_dimensionless = np.fft.rfftfreq(len(signal))
T = dt*N
df = 1./T
fhz_nf = f_dimensionless*df*N

if q1:
    fig1 = plt.figure(figsize=(14,4.5))
    plt.title('Q1 (a)')
    plt.xlabel('tempo [s]')
    plt.ylabel('sinal ECG não filtrado [mV]')
    plt.plot(t,signal)
    plt.grid(b=True,which='both')
    plt.tight_layout()
    plt.savefig('Q1_a_sinal_nfilt.png')
    
    fig2 = plt.figure(figsize=(14,4.5))
    plt.title('Q1 (b)')
    plt.xlabel(u'frequência [Hz]')
    plt.ylabel(u'FFT do sinal ECG não filtrado')
    plt.plot(fhz_nf,sig_fft)
    plt.tight_layout()
    plt.savefig('Q1_b_espectro_nfilt.png')

# Questão 2
# (c)
notch_f = 50.3
fny = fs/2.

notch_b, notch_a = sgn.iirnotch(notch_f/fny, 10)
notch_w, notch_h = sgn.freqz(notch_b, notch_a, worN=len(fhz_nf),fs=fs*2*np.pi)

sig_notch = sgn.lfilter(notch_b,notch_a,signal)

sig_fft_notch = np.fft.rfft(sig_notch)
f_dimensionless_notch = np.fft.rfftfreq(len(sig_notch))
T = dt*N
df = 1./T
fhz_notch = f_dimensionless_notch*df*N

if q2:
    fig3 = plt.figure(figsize=(14,4.5))
    plt.title('Q2 (c)')
    plt.xlabel('tempo [s]')
    plt.ylabel('sinal ECG filtrado Notch [mV]')
    plt.plot(t,sig_notch)
    plt.grid(b=True,which='both')
    plt.tight_layout()
    plt.savefig('Q2_c_sinal_filt_Notch.png')
    
    fig4 = plt.figure(figsize=(14,4.5))
    plt.title('Q2 (c)')
    plt.xlabel(u'frequência [Hz]')
    plt.ylabel('FFT sinal filtrado Notch ')
    plt.plot(fhz_notch,sig_fft_notch)
    plt.tight_layout()
    plt.savefig('Q2_c_espectro_filt_Notch.png')

    fig5 = plt.figure()
    plt.plot(notch_w/(2*np.pi), 20 * np.log10(abs(notch_h)))
    plt.xscale('log')
    plt.title('Filtro Notch')
    plt.xlabel(u'Frequência [Hz]')
    plt.ylabel(u'Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    #plt.axvline(notch_fpass, color='green') # cutoff frequency
    plt.savefig('Q2_c_Notch_respfreq.png')

'''
# (b)
hp_fpass = .7
hp_fstop = .5
fny = fs/2.
#wpass = fpass*2*np.pi
#wstop = fstop*2*np.pi
#wny = fny*2*np.pi

hp_ord, hp_wn = sgn.buttord(hp_fpass/fny,hp_fstop/fny,3,40)

#b, a = sgn.butter(filt_ord, filt_wn, 'low')
hp_b, hp_a = sgn.butter(4, hp_fpass/fny, 'highpass')
hp_w, hp_h = sgn.freqz(hp_b, hp_a, worN=len(fhz_lp),fs=fs*2*np.pi)

fig6 = plt.figure()
plt.plot(hp_w/(2*np.pi), 20 * np.log10(abs(hp_h)))
plt.xscale('log')
plt.title('Filtro Butterworth PA')
plt.xlabel(u'Frequência [Hz]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(hp_fpass, color='green') # cutoff frequency

sig_lphp= sgn.lfilter(hp_b,hp_a,sig_lp)

sig_fft_lphp = np.fft.rfft(sig_lphp)
f_dimensionless_lphp = np.fft.rfftfreq(len(sig_lphp))
T = dt*N
df = 1./T
fhz_lphp = f_dimensionless_lphp*df*N

if q2:
    fig7 = plt.figure(figsize=(14,4.5))
    plt.title('Q2 (b)')
    plt.xlabel('tempo [s]')
    plt.ylabel('sinal ECG filtrado PB + PA [mV]')
    plt.plot(t,sig_lphp)
    plt.grid(b=True,which='both')
    plt.tight_layout()
    plt.savefig('Q2_b_sinal_filt_PBPA.png')
    
    fig8 = plt.figure(figsize=(14,4.5))
    plt.title('Q2 (b)')
    plt.xlabel(u'frequência [Hz]')
    plt.ylabel('FFT sinal filtrado PB + PA')
    plt.plot(fhz_lphp,sig_fft_lphp)
    plt.tight_layout()
    plt.savefig('Q2_b_espectro_filt_PBPA.png')
'''
