# prepare the input and labels for traing and testing of the NN network
import numpy as np
from utils import *
import torch
from torch.utils import data
import time
import os


class UWADNNDataset(data.Dataset):
    def __init__(self, CIR_dir, idx_low, idx_high, transform=None):
        # load CIR_dataset
        self.channel_response_set, h_line = load_channel_mat(CIR_dir, idx_low, idx_high)
        Pilot_file_name = 'Pilot_' + str(P)
        if os.path.isfile(Pilot_file_name):
            print('Load Training Pilots txt')
            bits = np.loadtxt(Pilot_file_name, delimiter=',')
        else:
            bits = np.random.binomial(n=1, p=0.5, size=(K * mu,))
            np.savetxt(Pilot_file_name, bits, delimiter=',')

        self.pilotValue = Modulation(bits, mu)

    def __len__(self):
        return len(self.channel_response_set)

    def __getitem__(self, idx):

        # Hf_ls, Yf, Yf_EQ as the input for TaskNet
        # Hf, Sf, Xbit as the label for TaskNet

        # input for FNN net:received Yt+Pt
        # label for FNN net: transmitted bit [16:32]
        allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
        if P < K:
            pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
            dataCarriers = np.delete(allCarriers, pilotCarriers)

        else:  # K = P
            pilotCarriers = allCarriers
            dataCarriers = []

        payloadBits_per_OFDM = K * mu
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        channelResponse = self.channel_response_set[idx]

        signal_output = ofdm_simulate(bits, channelResponse, SNR, mu, with_CP_flag, K, P, CP, self.pilotValue,
                                            pilotCarriers, dataCarriers,
                                            Clipping_Flag, nnMode='DNN')
        return torch.as_tensor(signal_output, dtype=torch.float), torch.as_tensor(bits[16:32], dtype=torch.float)


class UWAComDataset(data.Dataset):
    def __init__(self, CIR_dir, idx_low, idx_high, modem, transform=None):
        # load CIR_dataset
        self.channel_response_set, h_line = load_channel_mat(CIR_dir, idx_low, idx_high)
        Pilot_file_name = 'Pilot_' + str(P)+'_'+str(mu)
        if os.path.isfile(Pilot_file_name):
            print('Load Training Pilots txt')
            bits = np.loadtxt(Pilot_file_name, delimiter=',')
        else:
            bits = np.random.binomial(n=1, p=0.5, size=(K * mu,))
            np.savetxt(Pilot_file_name, bits, delimiter=',')

        self.pilotValue = Modulation(bits, mu)
        self.modem = modem

    def __len__(self):
        return len(self.channel_response_set)

    def __getitem__(self, idx):
        allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
        if P < K:
            pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
            dataCarriers = np.delete(allCarriers, pilotCarriers)

        else:  # K = P
            pilotCarriers = allCarriers
            dataCarriers = []

        payloadBits_per_OFDM = K * mu
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        channelResponse = self.channel_response_set[idx]
        # generate data
        signal_output = ofdm_simulate(self.modem, bits, channelResponse, SNR, mu, with_CP_flag, K, P, CP, self.pilotValue,
                                            pilotCarriers, dataCarriers,
                                            Clipping_Flag, nnMode='ComNet')
        Yd, h_ls, true_H = signal_output
        return torch.as_tensor(Yd, dtype=torch.float), \
               torch.as_tensor(h_ls, dtype=torch.float), \
               torch.as_tensor(true_H, dtype=torch.float),\
               torch.as_tensor(bits[96:144], dtype=torch.float)  # transmitted bits as the label of final loss


if __name__ == '__main__':
    modem = QAMModem(K)
    time_start = time.time()

    #trainset = UWADNNDataset(CIR_dir=Train_set_path, idx_low=1, idx_high=300)
    trainset = UWAComDataset(CIR_dir=Train_set_path, idx_low=1, idx_high=100, modem=modem)
    trainloader = data.DataLoader(trainset, 64, shuffle=True,num_workers=8, pin_memory=True)
    time_end = time.time()
    time_elasped = time_end-time_start
    print('Training set: %6d samples, %4d batches' % (len(trainset), len(trainloader)))
    print('Time for loading the datset is: %4f sec' % time_elasped)

