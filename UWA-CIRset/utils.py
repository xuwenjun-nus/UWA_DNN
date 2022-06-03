import numpy as np
# import scipy.interpolate
# import tensorflow as tf
import math
import os
#from load_channel import load_channel_mat
from load_channel import load_channel_mat
import time
from numpy.linalg import inv
from commpy.modulation import QAMModem
#from deployment import BER

# global defs
P = 64     # number of pilot carriers per ofdm block
with_CP_flag = True
SNR = 10
Clipping_Flag = False
K = 64 # number of subcarriers
CP = K // 4 # length of CP
mu = 2 # modulation order
mu = 6 # for 64QAM
#Train_set_path ='./SIM-P/Rayleigh'
Train_set_path = '/home/pcl2/ofdm_nn/OFDM_DNN-master/H_dataset/H_dataset/'
Test_set_path ='./CWR/CWR/WL'
model_path = './checkpoints_FC/best_0.0025_200.pth'
LOG_DIR = './tb'
CR = 1

# mapping table for QPSK
mapping_table = {
    (0,0) : -1-1j,
    (0,1) : -1+1j,
    (1,0) : 1-1j,
    (1,1) : 1+1j,
}

decoding = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1)
}

demapping_table = {v : k for k, v in mapping_table.items()}

# SNR coefficient, related to modulation order
beta = {
    2: 1,
    3: 1/9,
    4: 17/9,
    5: 1.089,
    6: 2.36
}


def print_something():
    print('utils.py has been loaded perfectly')


def BER(A, B):
    """
    :param A: bits 1-d array
    :param B: bits 1-d array
    :return: BER
    """
    id = np.count_nonzero(np.abs(A-B))
    return id/len(A)


def Clipping(x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL * sigma
    x_clipped = x
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx] * CL), abs(x_clipped[clipped_idx]))
    return x_clipped


def PAPR(x):
    Power = np.abs(x) ** 2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10 * np.log10(PeakP / AvgP)
    return PAPR_dB


def Modulation(bits, mu):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return (2 * bit_r[:, 0] - 1) + 1j * (2 * bit_r[:, 1] - 1)  # This is just for QAM modulation

# this is for QAM demodulation
#def Demodulation(symbol, mu):

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time, CP, CP_flag, mu, K):
    if CP_flag == False:
        # add noise CP
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K * mu,))
        codeword_noise = Modulation(bits_noise, mu)
        OFDM_data_nosie = codeword_noise
        OFDM_time_noise = np.fft.ifft(OFDM_data_nosie)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]  # take the last CP samples ...
    # cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved ** 2))
    sigma2 = signal_power * 10 ** (-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise


def removeCP(signal, CP, K):
    return signal[CP:(CP + K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest


def get_payload(equalized):
    return equalized[dataCarriers]


def PS(bits):
    return bits.reshape((-1,))


def ofdm_simulate(modem, codeword, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers,
                  Clipping_Flag, nnMode= None):
    payloadBits_per_OFDM = mu * len(dataCarriers)
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        #QAM = Modulation(bits, mu)
        QAM = modem.modulate(bits)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue

    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, K)
    # OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX, CR)  # add clipping
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP, K)
    # OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    #codeword_qam = Modulation(codeword, mu)
    codeword_qam = modem.modulate(codeword)
    if len(codeword_qam) != K:
        print('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    # OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    if Clipping_Flag:
        OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword, CR)  # add clipping
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword, CP, K)

    # OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    if nnMode =='DNN':
        signal_output = np.concatenate((np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP))),
                           np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword))))), abs(
            channelResponse)
    else:
        Yd = DFT(OFDM_RX_noCP_codeword) # received symbol in frequency domain
        # use FFT to move to frequency domain
        Pilot_RX_F = DFT(OFDM_RX_noCP)
        h_ls = Pilot_RX_F / pilotValue
        true_H = np.fft.fft(channelResponse, n=K)
        signal_output = (np.concatenate((np.real(Yd), np.imag(Yd))), \
                         np.concatenate((np.real(h_ls), np.imag(h_ls))), \
                         np.concatenate((np.real(true_H), np.imag(true_H))))

    return signal_output


def LS(input, output):
    """

    :param input: the input pilot signal P--> complex
    :param output: teh received signal Y --> complex
    :return: h_ls, the estimated h with least square
    """

    assert np.iscomplex(input), 'The type of pilot signal should be complex!'
    assert np.iscomplex(output), 'The type of received signal should be complex!'

    return output/input


def H_MMSE(Y, Xp,Nfft, Nps, h , SNR):
    """
    
    :param Y: Received signal in frequency domain
    :param Xp: The transmitted pilot
    :param Nfft: FFT size
    :param Nps: Pilot spacing
    :param h: channel impulse response 
    :param SNR: 
    :return: h_mmse
    """
    snr = np.power(10, SNR * 0.1)
    Np = Nfft // Nps
    k = np.arange(Np)
    H_tilte = Y[0:Nfft:Nps]/Xp
    #H_tilde = Y[0, pilot_loc[:Np]]/ Xp # H_LS
    #k = 0:length(h) - 1
    k = np.arange(len(h))
    #hh = np.absolute(h) # total power of h
    hh = np.dot(h, h.conjugate())
    tmp = h * h.conjugate() * k
    df = 1 / Nfft
    r = sum(tmp) / hh
    r2 = np.dot(tmp, k) /hh
    tau_rms = np.sqrt(np.real(r2-r*r)) # rms delay
    df = 1 / Nfft
    j2pi_tau_df = 1j * 2 * np.pi * tau_rms * df
    k1 = np.arange(Nfft)
    K1 = np.tile(k1,(Np, 1)).transpose()
    K2 = np.tile(np.arange(Np), (Nfft, 1))
    rf = 1/ (1 + j2pi_tau_df * (K1 - Nps * K2))
    K3 = np.tile(np.arange(Np), (Np, 1)).transpose()
    K4 = np.tile(np.arange(Np), (Np, 1))
    rf2 = 1/ (1 + j2pi_tau_df * Nps * (K3 - K4))
    Rhp = rf
    Rpp = rf2 + np.identity(H_tilte.shape[0])/snr
    H_est = np.matmul(Rhp, inv(Rpp))
    W = H_est    # W: linear matrix
    H_est = np.matmul(H_est, H_tilte).transpose()

    return H_est


def H_LMMSE(Y, Xp,Nfft, Nps, h , SNR, mu):
    """
    :param Y:  Received signal in frequency domain
    :param Xp: The transmitted pilot
    :param Nfft: FFT size
    :param Nps: Pilot spacing
    :param h: channel impulse response
    :param SNR: signal to noise ratio (in dB)
    :param mu: modulation order
    :return: H_est estimated channel response and W: weight matrix
    """
    b = beta[mu]
    snr = np.power(10, SNR * 0.1)
    Np = Nfft // Nps
    k = np.arange(Np)
    H_tilte = Y[0:Nfft:Nps] / Xp
    # H_tilde = Y[0, pilot_loc[:Np]]/ Xp # H_LS
    # k = 0:length(h) - 1
    k = np.arange(len(h))
    # hh = np.absolute(h) # total power of h
    hh = np.dot(h, h.conjugate())
    tmp = h * h.conjugate() * k
    df = 1 / Nfft
    r = sum(tmp) / hh
    r2 = np.dot(tmp, k) / hh
    tau_rms = np.sqrt(np.real(r2 - r * r))  # rms delay
    df = 1 / Nfft
    j2pi_tau_df = 1j * 2 * np.pi * tau_rms * df
    k1 = np.arange(Nfft)
    K1 = np.tile(k1, (Np, 1)).transpose()
    K2 = np.tile(np.arange(Np), (Nfft, 1))
    rf = 1 / (1 + j2pi_tau_df * (K1 - Nps * K2))
    K3 = np.tile(np.arange(Np), (Np, 1)).transpose()
    K4 = np.tile(np.arange(Np), (Np, 1))
    rf2 = 1 / (1 + j2pi_tau_df * Nps * (K3 - K4))
    Rhp = rf
    Rpp = rf2 + b * np.identity(H_tilte.shape[0]) / snr
    W = np.matmul(Rhp, inv(Rpp)) # weight matrix
    H_est = np.matmul(W, H_tilte).transpose()
    return H_est, W


def ofdm_simulate_H(modem, codeword, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers,
                  Clipping_Flag):
    payloadBits_per_OFDM = mu * len(dataCarriers)
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        #QAM = Modulation(bits, mu)
        QAM = modem.modulate(bits.astype(int))
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue

    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, K)
    # OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX, CR)  # add clipping
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP, K)

    # OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    #codeword_qam = Modulation(codeword, mu)
    codeword_qam = modem.modulate(codeword)

    if len(codeword_qam) != K:
        print('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    # OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    if Clipping_Flag:
        OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword, CR)  # add clipping
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword, CP, K)
    # OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    #recover signal with H_hat
    start_t = time.time()
    codeword_RX_F = DFT(OFDM_RX_noCP_codeword) # in frequency domain
    # use FFT to move to frequency domain
    Pilot_RX_F = DFT(OFDM_RX_noCP)
    h_ls = Pilot_RX_F / OFDM_data
    # channel estimation with h_hat
    symbol_RX = PS(codeword_RX_F/h_ls)
    # demodulation
    # demap = lambda x: np.argmin([np.absolute(x-mapping_table[(0, 0)]), np.absolute(x-mapping_table[(0, 1)]),\
    #                             np.absolute(x-mapping_table[(1, 0)]), np.absolute(x-mapping_table[(1, 1)])])
    # bits_demap = [demap(ii) for ii in symbol_RX]
    # bits_recover = np.array([decoding[jj] for jj in bits_demap])
    # bits_recover_ls = bits_recover.reshape(len(bits_demap)*2)
    bits_recover_ls = modem.demodulate(symbol_RX, demod_type='hard')
    elapsed_ls = time.time() - start_t


    # channel estimation with mmse
    start = time.time()
    codeword_RX_F = DFT(OFDM_RX_noCP_codeword)
    # use FFT to move to frequency domain
    Pilot_RX_F = DFT(OFDM_RX_noCP)
    h_mmse = H_MMSE(Pilot_RX_F, OFDM_data, K, 1, channelResponse, SNR)
    h_lmmse, W = H_LMMSE(Pilot_RX_F, OFDM_data, K, 1, channelResponse, SNR, mu)
    symbol_RX = PS(codeword_RX_F / h_mmse)
    # bits_demap = [demap(ii) for ii in symbol_RX]
    # bits_recover = np.array([decoding[jj] for jj in bits_demap])
    #bits_recover_mmse = bits_recover.reshape(len(bits_demap) * 2)
    bits_recover_mmse = modem.demodulate(symbol_RX, demod_type='hard')
    elapsed_mmse = time.time() - start

    # true H and the corresponding recovered signal
    true_H = np.fft.fft(channelResponse, n=64)
    symbol_RX = PS(codeword_RX_F / true_H)
    bits_recover_true = modem.demodulate(symbol_RX, demod_type='hard')
    return h_ls, h_mmse, h_lmmse, W, bits_recover_ls, elapsed_ls,\
           bits_recover_mmse, elapsed_mmse, bits_recover_true # sparse_mask


def ofdm_simulate_CP(modem, codeword, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers,
                  Clipping_Flag):
    " CP removal case"
    payloadBits_per_OFDM = mu * len(dataCarriers)
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        # QAM = Modulation(bits, mu)
        QAM = modem.modulate(bits.astype(int))
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_TX = OFDM_time
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX, CR)  # add clipping
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX = OFDM_RX[:K]

    symbol = np.zeros(K, dtype=complex)
    # codeword_qam = Modulation(codeword, mu)
    codeword_qam = modem.modulate(codeword)

    if len(codeword_qam) != K:
        print('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    #OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    if Clipping_Flag:
        OFDM_time_codeword = Clipping(OFDM_time_codeword, CR)  # add clipping
    OFDM_RX_codeword = channel(OFDM_time_codeword, channelResponse, SNRdb)
    OFDM_RX_codeword = OFDM_RX_codeword[:K]
    #OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword, CP, K
    # recover signal with H_hat
    start_t = time.time()
    codeword_RX_F = DFT(OFDM_RX_codeword)  # in frequency domain
    # use FFT to move to frequency domain
    Pilot_RX_F = DFT(OFDM_RX)
    #channel estimation with LS
    h_ls = Pilot_RX_F / OFDM_data
    symbol_RX = PS(codeword_RX_F / h_ls) # zero forcing

    bits_recover_ls = modem.demodulate(symbol_RX, demod_type='hard')
    elapsed_ls = time.time() - start_t

    # channel estimation with mmse
    start = time.time()
    codeword_RX_F = DFT(OFDM_RX_codeword)
    # use FFT to move to frequency domain
    Pilot_RX_F = DFT(OFDM_RX)
    #h_mmse = H_MMSE(Pilot_RX_F, OFDM_data, K, 1, channelResponse, SNR)
    h_lmmse, W = H_LMMSE(Pilot_RX_F, OFDM_data, K, 1, channelResponse, SNR, mu)
    symbol_RX = PS(codeword_RX_F / h_mmse)
    # bits_demap = [demap(ii) for ii in symbol_RX]
    # bits_recover = np.array([decoding[jj] for jj in bits_demap])
    # bits_recover_mmse = bits_recover.reshape(len(bits_demap) * 2)
    bits_recover_mmse = modem.demodulate(symbol_RX, demod_type='hard')
    elapsed_mmse = time.time() - start

    # true H and the corresponding recovered signal
    true_H = np.fft.fft(channelResponse, n=64)
    symbol_RX = PS(codeword_RX_F / true_H)
    bits_recover_true = modem.demodulate(symbol_RX, demod_type='hard')
    return h_ls, h_mmse, h_lmmse, W, bits_recover_ls, elapsed_ls, \
           bits_recover_mmse, elapsed_mmse, bits_recover_true  # sparse_mask


def weight_init():
    modem = QAMModem(K)
    allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
    if P < K:
        pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
        dataCarriers = np.delete(allCarriers, pilotCarriers)

    else:  # K = P
        pilotCarriers = allCarriers
        dataCarriers = []

    payloadBits_per_OFDM = K * mu

    SNRdb = SNR  # signal to noise-ratio in dB at the receiver
    # Clipping_Flag = Clipping

    Pilot_file_name = 'Pilot_' + str(P) + '_' + str(mu)
    if os.path.isfile(Pilot_file_name):
        print('Load Training Pilots txt')
        bits = np.loadtxt(Pilot_file_name, delimiter=',')
    else:
        bits = np.random.binomial(n=1, p=0.5, size=(K * mu,))
        np.savetxt(Pilot_file_name, bits, delimiter=',')

    pilotValue = modem.modulate(bits.astype(int))

    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
    # load channel response
    filepath = Train_set_path
    channel_response_set, h_line = load_channel_mat(filepath, 1, 61)
    CP_flag = with_CP_flag
    # perform lmmse estimation for 10 times and compute the average
    W = np.zeros((K, K), dtype='complex128')
    for i in range(10):
        channelResponse = channel_response_set[np.random.randint(0,len(channel_response_set))]
        h_ls, h_mmse, h_lmmse, W, bits_recover_ls, elapsed_ls, \
        bits_recover_mmse, elapsed_mmse, bits_recover_true = ofdm_simulate_H(modem, bits,
                                                                             channelResponse,
                                                                             SNRdb, mu,
                                                                             CP_flag, K,
                                                                             P, CP,
                                                                             pilotValue,
                                                                             pilotCarriers,
                                                                             dataCarriers,
                                                                             Clipping_Flag)

        W += W
    W = W/10
    return np.block([[np.real(W), -np.imag(W)], [np.imag(W), np.real(W)]])


if __name__ == '__main__':
    # try simulating ofdm
    # test H_mmse
    modem = QAMModem(K)
    # bits = [0, 1, 0, 1, 1, 0]
    # symbol = modem.modulate(bits)
    # x = modem.demodulate(symbol, demod_type='hard')
    Y = np.array([-1.2441 - 1j*1.1780, 0.1900 + 1j*2.0971, 0.5009 + 1j*2.2035, 0.6834 + 1j*2.3317, \
                  -2.3185 - 1j*0.9073, 1.2385 + 1j*2.1114, 1.4093 + 1j*1.8946, 1.4326 + 1j*1.6353, \
                  1.9793 - 1j*0.0376,  1.4082 + 1j*1.1298, 1.1063 + 1j*0.8667, 0.8113 + 1j*0.8814, \
                  1.0171 + 1j*0.2511,  0.2998 + 1j*1.0361, 0.1714 + 1j*1.2523, 0.0986 + 1j*1.5259])
    Xp = np.array([1, 1, -1, -1])
    pilot_loc = np.array([1, 5, 9, 13])
    Nfft = 16
    Nps = 4
    h = np.array([-1.6481 - 1j*0.5727,   0.3296 - 1j*0.6402])
    SNR = 10
    H_mmse = H_MMSE(Y, Xp, Nfft, Nps, h, SNR)
    allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
    if P < K:
        pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
        dataCarriers = np.delete(allCarriers, pilotCarriers)

    else:  # K = P
        pilotCarriers = allCarriers
        dataCarriers = []

    payloadBits_per_OFDM = K * mu

    SNRdb = SNR  # signal to noise-ratio in dB at the receiver
    #Clipping_Flag = Clipping

    Pilot_file_name = 'Pilot_' + str(P)+'_'+str(mu)
    if os.path.isfile(Pilot_file_name):
        print('Load Training Pilots txt')
        bits = np.loadtxt(Pilot_file_name, delimiter=',')
    else:
        bits = np.random.binomial(n=1, p=0.5, size=(K * mu,))
        np.savetxt(Pilot_file_name, bits, delimiter=',')

    pilotValue = Modulation(bits, mu)
    pilotSymbol = modem.modulate(bits.astype(int))

    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
    # load channel response
    filepath = Train_set_path
    channel_response_set, h_line = load_channel_mat(filepath, 1, 61)
    channelResponse = channel_response_set[0]
    CP_flag = with_CP_flag
    true_H = np.fft.fft(channelResponse, n=64)
    #signal_output, para = ofdm_simulate(bits, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers,
                 # Clipping_Flag)

    h_ls, h_mmse, h_lmmse, W, bits_recover_ls, elapsed_ls, bits_recover_mmse, elapsed_mmse, bits_recover_true = \
        ofdm_simulate_H(modem, bits, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers,
                                        dataCarriers,
                                        Clipping_Flag)
    print("Inference time of LS for one sample: %4.3f seconds！" % (elapsed_ls))
    # calculate BER
    ber_LS = BER(bits, bits_recover_ls)
    ber_MMSE = BER(bits, bits_recover_mmse)
    ber_true = BER(bits, bits_recover_true)

    h_ls, h_mmse, h_lmmse, W, bits_recover_ls, elapsed_ls, bits_recover_mmse, elapsed_mmse, bits_recover_true = \
        ofdm_simulate_H(modem, bits, channelResponse, SNRdb, mu, CP_flag, K, P, 8, pilotValue, pilotCarriers,
                        dataCarriers,
                        Clipping_Flag=True)

    # calculate BER
    ber_LS_CP = BER(bits, bits_recover_ls)
    ber_MMSE_CP = BER(bits, bits_recover_mmse)
    ber_true_CP = BER(bits, bits_recover_true)

   # W2 = weight_init()



