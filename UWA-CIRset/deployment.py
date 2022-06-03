import torch
from model import FC_Net
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


def deploy(args, net, valloader):
    net.eval()
    ber = 0
    with torch.no_grad():
        for _, (input, gt) in enumerate(valloader):
            input= input.to(args.device, non_blocking=True)

            output = net(input)
            # the next two lines are for KL divergence
            output = output.view(-1, output.size()[-1])
            gt = gt.view(-1, gt.size()[-1])
            ber += BER(gt, output.cpu().numpy())
    return ber/len(valloader)


def plot_BER(ber_mean, std_ber, snr, color, algo):
    # plot mean and std as error
    error = np.array(std_ber)**2
    plt.errorbar(snr, ber_mean, yerr=error, fmt='-o', color=color, label =algo)


def main():
    args = argument()
    cudnn.benchmark = True
    #testset = UWADNNDataset(CIR_dir=Train_set_path, idx_low=301, idx_high=400)
    #testloader = data.DataLoader(testset, 1, shuffle=False,
    #                             num_workers=args.num_worker, pin_memory=True)
    #print('Testing set: %6d samples, %4d batches' %
    #      (len(testset), len(testloader)))

    net = FC_Net(256, 16)
    net.load_state_dict(torch.load(model_path), strict=True)
    print('Loading model: %s' % (model_path))

    args.no_cuda = args.no_cuda or not cuda.is_available()
    if not args.no_cuda:
        args.device = torch.device('cuda:0')
        net = net.to(args.device)
    else:
        args.device = torch.device('cpu')
        net = net.to(args.device)

    print('Testing start')
    net.eval()
    Pilot_file_name = 'Pilot_' + str(P)
    if os.path.isfile(Pilot_file_name):
        print('Load Training Pilots txt')
        bits = np.loadtxt(Pilot_file_name, delimiter=',')
    else:
        bits = np.random.binomial(n=1, p=0.5, size=(K * mu,))
        np.savetxt(Pilot_file_name, bits, delimiter=',')

    pilotValue = Modulation(bits, mu)
    channel_response_set, h_line = load_channel_mat(Train_set_path, 301, 308)
    print('Testing set: %6d samples' %(h_line))
    payloadBits_per_OFDM = K * mu
    allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
    if P < K:
        pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
        dataCarriers = np.delete(allCarriers, pilotCarriers)

    else:  # K = P
        pilotCarriers = allCarriers
        dataCarriers = []
    print_freq = 10000
    # compute BER under various SNRs
    snr = np.arange(0, 35, 5)
    ber_mean = []
    ber_std = []
    ber_LS_mean, ber_LS_std = [], []
    ber_MMSE_mean, ber_MMSE_std = [], []
    for s in snr:
        ber = []
        ber_LS = []
        ber_MMSE = []
        for i in range(h_line):
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            channelResponse= channel_response_set[i]
            signal_output, para = ofdm_simulate(bits, channelResponse, s, mu, with_CP_flag, K, P, CP, pilotValue,
                                                pilotCarriers, dataCarriers,
                                                Clipping_Flag)
            input = torch.as_tensor(signal_output, dtype=torch.float)
            input = input.to(args.device, non_blocking=True)
            gt = bits[16:32]
            start = time.time()
            output = net(input)
            output = output.cpu().detach().numpy()
            # should do decoding
            output = np.where(output < 0.5, 0, 1)
            ber.append(BER(gt, output))
            end = time.time()-start

            # compute BER for LS method
            h_ls, h_mmse, bits_recover_ls, elapsed_ls, bits_recover_mmse, elapsed_mmse\
                = ofdm_simulate_H(bits, channelResponse, s, mu, with_CP_flag, K, P, CP, pilotValue,
                                                pilotCarriers, dataCarriers,
                                                Clipping_Flag)
            # compute BER for MMSE method

            ber_LS.append(BER(gt, bits_recover_ls[16:32]))
            ber_MMSE.append(BER(gt, bits_recover_mmse[16:32]))

            if i % print_freq == 0:
                print("Inference time of DNN for one sample: %7.6f seconds！" % end)
                print("Inference time of LS for one sample: %7.6f seconds！" % elapsed_ls)
                print("Inference time of MMSE for one sample: %7.6f seconds！" % elapsed_mmse)

        print('Testing for snr: %d finished!' % s)
        print("BER of DNN for snr: %d is %4.3f" % (s, np.mean(np.array(ber))))
        print("BER of LS for snr: %d is %4.3f" % (s, np.mean(np.array(ber_LS))))
        print("BER of MMSE for snr: %d is %4.3f" % (s, np.mean(np.array(ber_MMSE))))

        ber_mean.append(np.mean(np.array(ber)))
        ber_std.append(np.std(np.array(ber)))
        ber_LS_mean.append(np.mean(np.array(ber_LS)))
        ber_LS_std.append(np.std(np.array(ber_LS)))
        ber_MMSE_mean.append(np.mean(np.array(ber_MMSE)))
        ber_MMSE_std.append(np.std(np.array(ber_MMSE)))

    # plotting
    fig = plt.figure(figsize=(16, 9))
    plt.rcParams.update({'font.size': 18})

    #plt.show()
    plot_BER(ber_mean, ber_std, snr, 'r', 'DNN')
    plot_BER(ber_LS_mean, ber_LS_std, snr, 'k', 'LS')
    plot_BER(ber_MMSE_mean, ber_MMSE_std, snr, 'b', 'MMSE')
    plt.xlabel('SNR(dB)', fontsize=18)
    plt.ylabel('BER', fontsize=18)
    plt.yscale('log')
    # plt.title('compare')
    plt.xticks(np.arange(0, np.max(snr) + 1, 10))
    # plt.legend()
    plt.grid(True, which='both', ls="-", color='0.65')
    plt.legend(loc='best')
    plt.savefig(args.save_ber)


if __name__ == '__main__':
    main()

