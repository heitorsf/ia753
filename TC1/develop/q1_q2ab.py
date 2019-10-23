import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import scipy.signal as sgn

q1 = True
q2 = True
PB_filt = True
PA_filt = False
Notch_filt = False

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

#if q1:
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
#plt.grid(b=True,which='both')
plt.plot(fhz_nf,sig_fft)
plt.tight_layout()
xlim_fft = plt.xlim()
ylim_fft = plt.ylim()
plt.savefig('Q1_b_espectro_nfilt.png')

# Questão 2
# (a)
if PB_filt:
    lp_fpass = 32.
    lp_fstop = 50.3
    fny = fs/2.
    #wpass = fpass*2*np.pi
    #wstop = fstop*2*np.pi
    #wny = fny*2*np.pi
    
    lp_ord, lp_wn = sgn.buttord(lp_fpass/fny,lp_fstop/fny,3,40)
    
    lp_b, lp_a = sgn.butter(lp_ord, lp_wn, 'low')
    #lp_b, lp_a = sgn.butter(8, lp_fpass/fny, 'low')
    lp_w, lp_h = sgn.freqz(lp_b, lp_a, worN=len(fhz_nf),fs=fs*2*np.pi)
    
    sig_lp = sgn.lfilter(lp_b,lp_a,signal)
    
    sig_fft_lp = np.fft.rfft(sig_lp)
    f_dimensionless_lp = np.fft.rfftfreq(len(sig_lp))
    T = dt*N
    df = 1./T
    fhz_lp = f_dimensionless_lp*df*N
    
    if q2:
        fig3 = plt.figure(figsize=(14,4.5))
        plt.title('Q2 (a)')
        plt.xlabel('tempo [s]')
        plt.ylabel('sinal ECG filtrado PB [mV]')
        plt.plot(t,sig_lp)
        plt.grid(b=True,which='both')
        plt.tight_layout()
        plt.savefig('Q2_a_sinal_filt_LP.png')
        
        fig4 = plt.figure(figsize=(14,4.5))
        plt.title('Q2 (a)')
        plt.xlabel(u'frequência [Hz]')
        plt.ylabel('FFT sinal filtrado PB')
        plt.xlim(xlim_fft)
        plt.ylim(ylim_fft)
        #plt.grid(b=True,which='both')
        plt.plot(fhz_lp,sig_fft_lp)
        plt.tight_layout()
        plt.savefig('Q2_b_espectro_filt_PB.png')
    
        fig5 = plt.figure()
        plt.plot(lp_w/(2*np.pi), 20 * np.log10(abs(lp_h)))
        plt.xscale('log')
        plt.title('Filtro Butterworth PB')
        plt.xlabel(u'Frequência [Hz]')
        plt.ylabel(u'Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(lp_fpass, color='green') # cutoff frequency
        plt.savefig('Q2_a3_PB_respfreq.png')

# (b)
if PA_filt:
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
