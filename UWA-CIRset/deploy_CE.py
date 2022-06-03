import torch
from model import CENet
from data_generator import UWADNNDataset
import numpy as np
from torch.backends import cudnn
from torch.utils import data
from config import argument
from torch import cuda
from utils import *
import time
import matplotlib.pyplot as plt
import os

algorithm = ['LMMSE', 'ComNet', 'LMMSE_CP', 'ComNet_CP', 'SameSNR']


def deploy(args, net, valloader, criterion):
    net.eval()
    ber = 0
    total_loss = 0
    with torch.no_grad():
        for _, (y_d, h_ls, true_H, bits) in enumerate(valloader):
            if args.model == 'CENet':
                input = h_ls
                gt = true_H
            else:
                input = (y_d, h_ls)
                gt = bits
            input= input.to(args.device, non_blocking=True)

            output = net(input)
            # the next two lines are for MSE error in db
            output = output.view(-1, output.size()[-1])
            gt = gt.view(-1, gt.size()[-1])
            loss = criterion(output, gt)
            total_loss += loss.item() * input.size()[0]
    return total_loss/len(valloader)


def plot_MSE(mse_mean, snr, color, algo):
    # plot mean and std as error
    plt.plot(snr, mse_mean, linestyle='--' if 'CP' in algo else '-',\
             marker='o' if 'CP' in algo else '^',\
             color=color, label=algo)


def main():
    args = argument()
    cudnn.benchmark = True
    modem = QAMModem(K)
    path = os.path.join(args.save_path,'CE_best_0.0064_1800.pth')
    W = torch.as_tensor(weight_init(), dtype=torch.float)
    net = CENet(CE_dim=128, W=W)
    net.load_state_dict(torch.load(path), strict=True)
    print('Loading model: %s' % (path))

    args.no_cuda = args.no_cuda or not cuda.is_available()
    if not args.no_cuda:
        args.device = torch.device('cuda:0')
        net = net.to(args.device)
    else:
        args.device = torch.device('cpu')
        net = net.to(args.device)

    print('Testing start')
    net.eval()
    Pilot_file_name = 'Pilot_' + str(P)+'_'+str(mu)
    if os.path.isfile(Pilot_file_name):
        print('Load Training Pilots txt')
        bits = np.loadtxt(Pilot_file_name, delimiter=',')
    else:
        bits = np.random.binomial(n=1, p=0.5, size=(K * mu,))
        np.savetxt(Pilot_file_name, bits, delimiter=',')

    pilotValue = modem.modulate(bits.astype(int))
    channel_response_set, h_line = load_channel_mat(Train_set_path, 1, 2)
    print('Testing set: %6d samples' % (h_line))
    payloadBits_per_OFDM = K * mu
    allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
    if P < K:
        pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
        dataCarriers = np.delete(allCarriers, pilotCarriers)

    else:  # K = P
        pilotCarriers = allCarriers
        dataCarriers = []
    print_freq = 10000
    # compute MSE  under various SNRs for the linear cases
    snr = np.arange(5, 42, 5)
    mse_LS_mean, mse_LMMSE_mean, mse_ComNet_mean = [], [], []
    mse_LS_cp_mean, mse_LMMSE_cp_mean, mse_ComNet_cp_mean = [], [], []
    for s in snr:
        mse_CE = 0.0
        mse_LS= 0.0
        mse_LMMSE = 0.0
        mse_CE_cp = 0.0
        mse_LS_cp = 0.0
        mse_LMMSE_cp = 0.0
        for i in range(h_line):
            #random transimitted signal
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))

            # the true_H
            channelResponse = channel_response_set[i]
            true_H = np.fft.fft(channelResponse, n=64)
            power = np.abs(true_H) ** 2 # power of true_H
            gt = np.concatenate((np.real(true_H), np.imag(true_H)))
            # ofdm linear case environment with CP
            h_ls, h_mmse, h_lmmse, W, bits_recover_ls, elapsed_ls, bits_recover_mmse, elapsed_mmse, bits_recover_true = \
                ofdm_simulate_H(modem, bits, channelResponse, s, mu, with_CP_flag, K, P, CP, pilotValue, pilotCarriers,
                        dataCarriers,
                        Clipping_Flag)

            input = torch.as_tensor(np.concatenate((np.real(h_ls), np.imag(h_ls))), dtype=torch.float)
            input = input.to(args.device, non_blocking=True)

            start = time.time()
            output = net(input)
            output = output.cpu().detach().numpy()

            # compute MSE for CENet
            mse = (np.square(gt - output)).mean()
            # convert to db
            msedB = 10 * np.log10(mse/np.mean(power))
            mse_CE += msedB

            # compute MSE for LS
            h_ls = np.concatenate((np.real(h_ls), np.imag(h_ls)))
            mse = (np.square(gt - h_ls)).mean()
            msedB = 10 * np.log10(mse / np.mean(power))
            mse_LS += msedB

            # compute MSE for LMMSE
            h_lmmse = np.concatenate((np.real(h_lmmse), np.imag(h_lmmse)))
            mse = (np.square(gt - h_lmmse)).mean()
            msedB = 10 * np.log10(mse / np.mean(power))
            mse_LMMSE += msedB

            end = time.time() - start

            # compute mse  for cp_removal cases
            h_ls, h_mmse, h_lmmse, W, bits_recover_ls, elapsed_ls, bits_recover_mmse, elapsed_mmse, bits_recover_true = \
                ofdm_simulate_H(modem, bits, channelResponse, s, mu, with_CP_flag, K, P, 0, pilotValue, pilotCarriers,
                                dataCarriers,
                                Clipping_Flag)

            # compute MSE for Comnet_cp method
            # have to retrain or not?
            input = torch.as_tensor(np.concatenate((np.real(h_ls), np.imag(h_ls))), dtype=torch.float)
            input = input.to(args.device, non_blocking=True)
            start = time.time()
            output = net(input) # the original model
            end = time.time()-start
            output = output.cpu().detach().numpy()

            # compute MSE for CENet
            mse = (np.square(gt - output)).mean()
            # convert to db
            msedB = 10 * np.log10(mse / np.mean(power))
            mse_CE_cp += msedB

            # compute MSE for LS
            h_ls = np.concatenate((np.real(h_ls), np.imag(h_ls)))
            mse = (np.square(gt - h_ls)).mean()
            msedB = 10 * np.log10(mse / np.mean(power))
            mse_LS_cp += msedB

            # compute MSE for LMMSE
            h_lmmse = np.concatenate((np.real(h_lmmse), np.imag(h_lmmse)))
            mse = (np.square(gt - h_lmmse)).mean()
            msedB = 10 * np.log10(mse / np.mean(power))
            mse_LMMSE_cp += msedB

            if i % print_freq == 0:
                print("Inference time of  ComNet for one sample: %7.6f seconds！" % end)
                print("Inference time of LS for one sample: %7.6f seconds！" % elapsed_ls)
                print("Inference time of LMMSE for one sample: %7.6f seconds！" % elapsed_mmse)

        print('Testing for snr: %d finished!' % s)
        print("MSE of CENet for snr: %d is %4.3f" % (s, mse_CE/h_line))
        print("MSE of LS for snr: %d is %4.3f" % (s, mse_LS/h_line))
        print("MSE of MMSE for snr: %d is %4.3f" % (s, mse_LMMSE/h_line))
        print("MSE of CE_cp for snr: %d is %4.3f" % (s, mse_CE_cp/h_line))
        print("MSE of LS_cp for snr: %d is %4.3f" % (s, mse_LS_cp / h_line))
        print("MSE of LMMSE_cp for snr: %d is %4.3f" % (s, mse_LMMSE_cp / h_line))
        mse_ComNet_mean.append(mse_CE/h_line)
        mse_LS_mean.append(mse_LS/h_line)
        mse_LMMSE_mean.append(mse_LMMSE/h_line)
        mse_ComNet_cp_mean.append(mse_CE_cp/h_line)
        mse_LS_cp_mean.append(mse_LS_cp / h_line)
        mse_LMMSE_cp_mean.append(mse_LMMSE_cp / h_line)

    # plotting
    fig = plt.figure(figsize=(16, 9))
    plt.rcParams.update({'font.size': 18})

    # plt.show()
    plot_MSE(mse_ComNet_mean, snr, 'r', 'ComNet')
    plot_MSE(mse_LS_mean, snr, 'k', 'LS')
    plot_MSE(mse_LMMSE_mean, snr, 'b', 'LMMSE')

    plot_MSE(mse_ComNet_cp_mean, snr, 'r', 'ComNet_CP')
    plot_MSE(mse_LS_cp_mean, snr, 'k', 'LS_CP')
    plot_MSE(mse_LMMSE_cp_mean, snr, 'b', 'LMMSE_CP')
    plt.xlabel('SNR(dB)', fontsize=18)
    plt.ylabel('MSE(dB)', fontsize=18)
    # plt.title('compare')
    plt.xticks(np.arange(0, np.max(snr) + 1, 10))
    #plt.yticks(np.arange(0, np.max(snr) + 1, 10))
    # plt.legend
    plt.legend(loc='best')
    plt.savefig('MSE_CE.png')


if __name__ == '__main__':
    main()


