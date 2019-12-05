import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from scipy.signal import buttord,butter,freqz,lfilter,iirnotch

# Filtros
PB_filt = True
PA_filt = False
Notch_filt = False

# Plots
plotQ1 = False
plotQ2a = True
plotQ2b = False
plotQ2c = True
plotQ2d = False

# 
#mysignal = np.loadtxt('../data/signal.txt')
#mysignal = np.zeros(120)
#mysignal[[0,49,101,118,119]] = [77.5,88.75,105.62,112.56,113.0]
#for i in range(1,len(mysignal)):
    #if mysignal[i]==0:
        #mysignal[i] = mysignal[i-1]
mysignal = np.concatenate((np.linspace(77.,88.,50,endpoint=False),
                           np.linspace(88.,105.,51,endpoint=False),
                           np.linspace(105.,112.,17,endpoint=False),
                           np.linspace(112.,113,2,endpoint=True)))
fs = 500  # Hz
dt = 1./fs # seconds
t = np.linspace(0,dt*mysignal.shape[0],mysignal.shape[0],endpoint=False)
N = len(mysignal)


# Questão 1
sig_fft = np.fft.rfft(mysignal)
f_dimensionless = np.fft.rfftfreq(len(mysignal))
T = dt*N
df = 1./T
fhz_nf = f_dimensionless*df*N

if plotQ1:
    fig1 = plt.figure(figsize=(14,4.5))
    plt.title('Q1 (a)')
    plt.xlabel('tempo [s]')
    plt.ylabel('sinal ECG não filtrado [V]')
    plt.plot(t,mysignal)
    plt.grid(b=True,which='both')
    plt.tight_layout()
    
    fig2 = plt.figure(figsize=(14,4.5))
    plt.title('Q1 (b)')
    plt.xlabel(u'frequência [Hz]')
    plt.ylabel(u'FFT do sinal ECG não filtrado')
    #plt.grid(b=True,which='both')
    plt.plot(fhz_nf,sig_fft)
    plt.tight_layout()
    xlim_fft = plt.xlim()
    ylim_fft = plt.ylim()

# Questão 2
# (a)
if PB_filt:
    lp_fpass = 50.
    lp_fstop = 70.3
    fny = fs/2.
    #wpass = fpass*2*np.pi
    #wstop = fstop*2*np.pi
    #wny = fny*2*np.pi
    
    lp_ord, lp_wn = buttord(lp_fpass/fny,lp_fstop/fny,3,40)
    lp_b, lp_a = butter(lp_ord, lp_wn, 'lowpass')
    #lp_b, lp_a = butter(8, lp_fpass/fny, 'lowpass')
    lp_w, lp_h = freqz(lp_b, lp_a, worN=len(fhz_nf),fs=fs*2*np.pi)
    
    sig_lp = lfilter(lp_b,lp_a,mysignal)
    
    sig_fft_lp = np.fft.rfft(sig_lp)
    f_dimensionless_lp = np.fft.rfftfreq(len(sig_lp))
    T = dt*N
    df = 1./T
    fhz_lp = f_dimensionless_lp*df*N
    
    if plotQ2a:
        fig3 = plt.figure(figsize=(14,4.5))
        plt.title('Q2 (a)')
        plt.xlabel('tempo [s]')
        plt.ylabel('sinal ECG filtrado PB [V]')
        plt.plot(t,sig_lp)
        plt.grid(b=True,which='both')
        plt.tight_layout()
        
        fig4 = plt.figure(figsize=(14,4.5))
        plt.title('Q2 (a)')
        plt.xlabel(u'frequência [Hz]')
        plt.ylabel('FFT sinal filtrado PB')
        #plt.xlim(xlim_fft)
        #plt.ylim(ylim_fft)
        #plt.grid(b=True,which='both')
        plt.plot(fhz_lp,sig_fft_lp)
        plt.tight_layout()
    
        fig5 = plt.figure(figsize=(8,4))
        plt.plot(lp_w/(2*np.pi), 20 * np.log10(abs(lp_h)))
        plt.xscale('log')
        plt.title('Filtro Butterworth PB')
        plt.xlabel(u'Frequência [Hz]')
        plt.ylabel(u'Amplitude [dB]')
        #plt.ylim((-200,50))
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(lp_fpass, color='green') # cutoff frequency

        fig6 = plt.figure()
        plt.subplot(2,1,1)
        plt.xlabel('Tempo [s]')
        plt.ylabel('Sinal ECG [V]')
        #plt.xlim((0,2))
        #plt.ylim((-0.3,1))
        plt.plot(t,mysignal,label='Original')
        plt.legend()
        #
        plt.subplot(2,1,2)
        plt.xlabel('Tempo [s]')
        plt.ylabel('Sinal ECG [V]')
        #plt.xlim((0,2))
        #plt.ylim((-0.3,1))
        plt.plot(t,sig_lp,label='Filtrado (PB)')
        plt.legend()
        #

    # (a.4)
    fig3 = plt.figure(figsize=(14,4.5))
    plt.xlabel('Tempo [s]')
    plt.ylabel('Sinal ECG [V]')
    plt.xlim((5.77,6.44))
    plt.ylim((-0.22,0.95))
    for cut_freq_lp in [22,32,42]:
        lp_b, lp_a = butter(lp_ord,cut_freq_lp/fny, 'low')
        sig_lp = lfilter(lp_b,lp_a,mysignal)
        plt.plot(t,sig_lp,label='Freq. corte: '+str(cut_freq_lp)+' Hz')
        plt.legend()

# (b) Passa-altas
if PA_filt:
    afiltrar_PA = mysignal
    hp_fpass = .5
    hp_fstop = .1
    fny = fs/2.
    #wpass = fpass*2*np.pi
    #wstop = fstop*2*np.pi
    #wny = fny*2*np.pi
    
    hp_ord, hp_wn = buttord(hp_fpass/fny,hp_fstop/fny,3,40)
    
    #hp_b, hp_a = butter(hp_ord, hp_wn, 'highpass')
    hp_b, hp_a = butter(4, hp_fpass/fny, 'highpass')
    hp_w, hp_h = freqz(hp_b, hp_a, worN=len(fhz_nf),fs=fs*2*np.pi)
    
    fig6 = plt.figure(figsize=(8,4))
    plt.plot(hp_w/(2*np.pi), 20 * np.log10(abs(hp_h)))
    plt.legend()
    plt.xscale('log')
    plt.title('Filtro Butterworth PA')
    plt.xlabel(u'Frequência [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.ylim(ymin=-80)
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(hp_fpass, color='green') # cutoff frequency
    
    sig_hp= lfilter(hp_b,hp_a,afiltrar_PA)
    
    sig_fft_hp = np.fft.rfft(sig_hp)
    f_dimensionless_hp = np.fft.rfftfreq(len(sig_hp))
    T = dt*N
    df = 1./T
    fhz_hp = f_dimensionless_hp*df*N
    
    if plotQ2b:
        fig7 = plt.figure(figsize=(12,4.5))
        plt.title('Q2 (b)')
        plt.subplot(1,2,1)
        plt.xlabel('tempo [s]')
        plt.ylabel('Sinal ECG não filtrado [V]')
        plt.plot(t,mysignal,label='Original')
        plt.grid(b=True,which='both')
        plt.ylim((-0.4,1.))
        plt.legend()
        #
        plt.subplot(1,2,2)
        plt.xlabel('tempo [s]')
        plt.ylabel('Sinal ECG filtrado PA [V]')
        plt.plot(t,sig_hp,label='Filtrado PA')
        plt.grid(b=True,which='both')
        plt.ylim((-0.4,1.))
        plt.legend()
        plt.tight_layout()
        
        fig8 = plt.figure(figsize=(10,4.5))
        plt.title('Q2 (b)')
        plt.subplot(2,1,1)
        plt.xlabel(u'frequência [Hz]')
        plt.ylabel('FFT sinal não filtrado')
        plt.xlim((0.,10))
        plt.ylim((-250,180))
        plt.plot(fhz_nf,sig_fft,label='Original')
        plt.legend()
        #
        plt.subplot(2,1,2)
        plt.xlabel(u'frequência [Hz]')
        plt.ylabel('FFT sinal filtrado PA')
        plt.xlim((0.,10))
        plt.ylim((-250,180))
        plt.plot(fhz_hp,sig_fft_hp,label='Filtrado PA')
        plt.legend()
        plt.tight_layout()

    # (b.4)
    fig9 = plt.figure(figsize=(14,4.5))
    plt.xlabel('Tempo [s]')
    plt.ylabel('Sinal ECG [V]')
    plt.xlim((5.77,6.44))
    plt.ylim((-0.22,0.95))
    for i,cut_freq_hp in enumerate([0.1,0.5,5.0]):
        hp_b, hp_a = butter(hp_ord,cut_freq_hp/fny, 'highpass')
        sig_hp = lfilter(hp_b,hp_a,mysignal)
        plt.plot(t,i+sig_hp,label='Freq. corte: '+str(cut_freq_hp)+' Hz')
        plt.legend()

# Questão 2 (c)
if Notch_filt:
    notch_f = 50.3
    fny = fs/2.
    
    notch_b, notch_a = iirnotch(notch_f/fny, 25)
    notch_w, notch_h = freqz(notch_b, notch_a, worN=len(fhz_nf),fs=fs*2*np.pi)
    
    sig_notch = lfilter(notch_b,notch_a,mysignal)
    
    sig_fft_notch = np.fft.rfft(sig_notch)
    f_dimensionless_notch = np.fft.rfftfreq(len(sig_notch))
    T = dt*N
    df = 1./T
    fhz_notch = f_dimensionless_notch*df*N
    
    if plotQ2c:
        fig9 = plt.figure(figsize=(14,4.5))
        plt.title('Q2 (c)')
        plt.xlabel('tempo [s]')
        plt.ylabel('sinal ECG filtrado Notch [mV]')
        #plt.grid(b=True,which='both')
        plt.plot(t,sig_notch)
        plt.grid(b=True,which='both')
        plt.tight_layout()
        
        fig10 = plt.figure(figsize=(14,4.5))
        plt.title('Q2 (c)')
        plt.xlabel(u'frequência [Hz]')
        plt.ylabel('FFT sinal filtrado Notch ')
        plt.plot(fhz_notch,sig_fft_notch)
        plt.tight_layout()
    
        fig11 = plt.figure(figsize=(8,5))
        plt.plot(notch_w/(2*np.pi), 20 * np.log10(abs(notch_h)))
        plt.xscale('log')
        plt.title('Filtro Notch')
        plt.xlabel(u'Frequência [Hz]')
        plt.ylabel(u'Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        #plt.axvline(notch_fpass, color='green') # cutoff frequency

        fig12 = plt.figure()
        plt.subplot(2,1,1)
        plt.xlabel('Tempo [s]')
        plt.ylabel('Sinal ECG [V]')
        plt.xlim((0.2,1.4))
        plt.ylim((-0.3,1))
        plt.plot(t,mysignal,label='Original')
        plt.legend()
        #
        plt.subplot(2,1,2)
        plt.xlabel('Tempo [s]')
        plt.ylabel('Sinal ECG [V]')
        plt.xlim((0.2,1.4))
        plt.ylim((-0.3,1))
        plt.plot(t,sig_notch,label='Filtrado (Notch)')
        plt.legend()
        #
