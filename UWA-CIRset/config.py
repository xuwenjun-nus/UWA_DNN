import argparse
import logging


def argument():
    parser = argparse.ArgumentParser(description='Network training')
    parser.add_argument('--num_worker', type=int, default=16,
                            help='workers to load data')
    parser.add_argument('--batch_size', type=int, default=1024,
                            help='batch size for training on all GPUs')
    parser.add_argument('--minibatch_size', type=int, default=50,
                        help='batch size for training on all GPUs')
    parser.add_argument('--gpus', type=str, default=None,
                            help='deprecated, use env var CUDA_VISIBLE_DEVICES instead '
                            '---GPU ids to use, None for all, comma separated---')
    parser.add_argument('--no_cuda', action='store_true',
                            help='disable CUDA, use CPU training')
    parser.add_argument('--num_epoch', type=int, default=350,
                            help='epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                            help='base learning rate (at epoch 0)')
    parser.add_argument('--lr_steps', type=str, default=[50, 100],
                            help='epochs before reducing learning rate, comma separated')
    parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum for optimizer')
    parser.add_argument('--wd', type=float, default=1e-4,
                            help='weight decay')
    parser.add_argument('--resume', type=str, default=None,
                            help='resume from checkpoint, None for default weight init')
    parser.add_argument('--resume_optim', type=str, default=None,
                            help='resume optimizer from checkpoint, will overwrite --lr, None for fresh init')
    parser.add_argument('--start_epoch', type=int, default=0,
                            help='epoch number to start, matter only if resume')
    parser.add_argument('--print_freq', type=int, default=10,
                            help='frequency in iterations to print running loss')
    parser.add_argument('--val_freq', type=int, default=200,
                            help='frequency in epochs to run validation')
    parser.add_argument('--save_freq', type=int, default=10,
                            help='frequency in epochs to save routine checkpoint')
    parser.add_argument('--save_path', type=str, default='./checkpoints_Com',
                          help='path to save checkpoints')
    parser.add_argument('--save_ber', type=str, default='ber_compare.png',
                        help='path to save ber plots')
    parser.add_argument('--model', type=str, default='CENet',
                        help='the type of model, choose from {FC_DNN, ComNet1, ComNet2, ComNet3, CENet }')
    args = parser.parse_args()
    return args
