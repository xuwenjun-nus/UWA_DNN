import numpy as np
import matplotlib.pyplot as plt


K = 64 # number of OFDM subcarriers
CP = K//4  # length of the cyclic prefix: 25% of the block
P = 8 # number of pilot carriers per OFDM block
pilotValue = 3+3j # The known value each pilot transmits
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.

# For convenience of channel estimation, let's make the last carriers also be a pilot
pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
P = P+1

# data carriers are all remaining carriers
dataCarriers = np.delete(allCarriers, pilotCarriers)

print ("allCarriers:   %s" % allCarriers)
print ("pilotCarriers: %s" % pilotCarriers)
print ("dataCarriers:  %s" % dataCarriers)
fig = plt.figure(figsize=(16, 4))
plt.rcParams.update({'font.size': 18})
plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
plt.show()

mu = 4 # bits per symbol (i.e. 16QAM)
payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol

mapping_table = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}

fig = plt.figure(2, figsize=(16, 16))
for b3 in [0, 1]:
    for b2 in [0, 1]:
        for b1 in [0, 1]:
            for b0 in [0, 1]:
                B = (b3, b2, b1, b0)
                Q = mapping_table[B]
                plt.plot(Q.real, Q.imag, 'bo')
                plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')
plt.grid(True)
plt.xticks(np.arange(0, 5, 10))
plt.yticks(np.arange(0, 5, 10))
plt.show()


demapping_table = {v : k for k, v in mapping_table.items()}
channelResponse = np.array([1, 0, 0.3+0.3j])  # the impulse response of the wireless channel
H_exact = np.fft.fft(channelResponse, K)
plt.figure(3, figsize=(16, 9))
plt.plot(allCarriers, abs(H_exact))
plt.show()
plt.xlabel('Subcarrier index', fontsize=18)
plt.ylabel('|H(f)|', fontsize=18)

SNRdb = 25  # signal to noise-ratio in dB at the receiver
bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
print("Bits count: ", len(bits))
print("First 20 bits: ", bits[:20])
print("Mean of bits (should be around 0.5): ", np.mean(bits))


def SP(bits):
    return bits.reshape((len(dataCarriers), mu))


bits_SP = SP(bits)
print("First 5 bit groups")
print(bits_SP[:5,:])


def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])


QAM = Mapping(bits_SP)
print("First 5 QAM symbols and bits:")
print(bits_SP[:5,:])
print(QAM[:5])





