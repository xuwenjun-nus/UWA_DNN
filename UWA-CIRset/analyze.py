from functools import partial
import sys
import os
import argparse
import time
import numpy as np
import torch
from torch import nn
from model import ComNet1, FC_Net, ComNet2, ComNet3
from ptflops import get_model_complexity_info


names = []
modules = []
layers = []


def dims(seq):
    if len(seq) == 0:
        return '1'
    seq = [str(i) for i in seq]
    return 'x'.join(seq)


def conns(size_in, size_out, kernel_size, stride=1, padding=0, dilation=1):
    ndim = len(size_out)
    if '__iter__' not in dir(kernel_size):
        kernel_size = [kernel_size] * ndim
    if '__iter__' not in dir(stride):
        stride = [stride] * ndim
    if '__iter__' not in dir(padding):
        padding = [padding] * ndim
    if '__iter__' not in dir(dilation):
        dilation = [dilation] * ndim

    cnts = []
    for i, o, k, s, p, d in zip(size_in, size_out, kernel_size, stride, padding, dilation):
        cnt = 0
        for x in range(k):
            l = (p - x * d) // s + ((p - x * d) % s != 0)
            l = max(l, 0)
            u = (i + p - x * d) // s + ((i + p - x * d) % s != 0)
            u = min(u, o)
            cnt += u - l
        cnts.append(cnt)

    return cnts


def hook(module, input, output):
    if not isinstance(module, (
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.Linear,
        nn.Dropout, nn.Dropout2d, nn.Dropout3d,
        nn.ReLU, nn.LeakyReLU, nn.PReLU,
        nn.Sigmoid, nn.LogSigmoid,
        nn.Tanh,
        nn.Softmin, nn.Softmax, nn.Softmax2d, nn.LogSoftmax
    )):
        return

    if not isinstance(output, torch.Tensor):
        output = output[0]

    size_in = input[0].size()
    size_out = output.size()

    ops = {
        'macc': 0,
        'cmp': 0,
        'add': 0,
        'div': 0,
        'exp': 0
    }
    mem = {
        'act': 0,
        'param': 0,
        'buf': 0
    }

    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if module.padding_mode == 'zeros':
            cnts = conns(size_in[2:], size_out[2:],
                         module.kernel_size, module.stride, module.padding, module.dilation)
            ops['macc'] = np.prod(cnts) * size_out[0] * size_out[1] * \
                size_in[1] // module.groups
        else:
            ops['macc'] = size_out.numel() * np.prod(module.kernel_size) * \
                size_in[1] // module.groups

        mem['act'] = size_out.numel()
        mem['param'] = module.weight.size().numel()
        if module.bias is not None:
            mem['param'] += module.bias.size().numel()

    elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        cnts = conns(size_out[2:], size_in[2:],
                     module.kernel_size, module.stride, module.padding, module.dilation)
        ops['macc'] = np.prod(cnts) * size_out[0] * size_out[1] * \
            size_in[1] // module.groups

        mem['act'] = size_out.numel()
        mem['param'] = module.weight.size().numel()
        if module.bias is not None:
            mem['param'] += module.bias.size().numel()

    elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
        cnts = conns(size_in[2:], size_out[2:],
                     module.kernel_size, module.stride, module.padding, module.dilation)
        ops['cmp'] = np.prod(cnts) * size_out[0] * size_out[1]

        mem['act'] = size_out.numel()

    elif isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
        cnts = conns(size_in[2:], size_out[2:],
                     module.kernel_size, module.stride, module.padding)
        ops['add'] = np.prod(cnts) * size_out[0] * size_out[1]
        ops['div'] = size_out.numel()

        mem['act'] = size_out.numel()

    elif isinstance(module, (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)):
        dim_in = np.array(size_in[2:])
        dim_out = np.array(size_out[2:])
        kernel_size = dim_in // dim_out + dim_in % dim_out
        ops['cmp'] = size_out.numel() * np.prod(kernel_size)

        mem['act'] = size_out.numel()

    elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
        dim_in = np.array(size_in[2:])
        dim_out = np.array(size_out[2:])
        kernel_size = dim_in // dim_out + dim_in % dim_out
        ops['add'] = size_out.numel() * np.prod(kernel_size)
        ops['div'] = size_out.numel()

        mem['act'] = size_out.numel()

    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        ops['macc'] = size_out.numel() + size_out[1]
        ops['add'] = 3 * size_out.numel()
        ops['div'] = 2 * size_out.numel() + size_out[1]
        ops['exp'] = size_out.numel()
        if module.affine:
            ops['macc'] += size_out.numel()
        if module.track_running_stats:
            ops['macc'] += 4 * size_out[1]

        mem['act'] = size_out.numel()
        if module.affine:
            mem['param'] = module.weight.size().numel()
            mem['param'] += module.bias.size().numel()
        if module.track_running_stats:
            mem['buf'] = module.running_mean.size().numel()
            mem['buf'] += module.running_var.size().numel()

    elif isinstance(module, (nn.Linear,)):
        size_in = torch.Size((size_in[0], size_in[-1]) + size_in[1:-1])
        size_out = torch.Size((size_out[0], size_out[-1]) + size_out[1:-1])

        ops['macc'] = size_out.numel() * size_in[1]

        mem['act'] = size_out.numel()
        mem['param'] = module.weight.size().numel()
        if module.bias is not None:
            mem['param'] += module.bias.size().numel()

    elif isinstance(module, (nn.Dropout,)):
        ops['macc'] = size_out.numel()
        ops['cmp'] = size_out.numel()
        # TODO: rng op = size_out.numel()

        if not module.inplace:
            mem['act'] = size_out.numel()

    elif isinstance(module, (nn.Dropout2d, nn.Dropout3d)):
        ops['macc'] = size_out.numel()
        ops['cmp'] = size_out[1]
        # TODO: rng op = size_out[1]

        if not module.inplace:
            mem['act'] = size_out.numel()

    elif isinstance(module, (nn.ReLU,)):
        ops['cmp'] = size_out.numel()

        if not module.inplace:
            mem['act'] = size_out.numel()

    elif isinstance(module, (nn.LeakyReLU,)):
        ops['macc'] = size_out.numel()
        ops['cmp'] = size_out.numel()

        if not module.inplace:
            mem['act'] = size_out.numel()

    elif isinstance(module, (nn.PReLU,)):
        ops['macc'] = size_out.numel()
        ops['cmp'] = size_out.numel()

        mem['act'] = size_out.numel()
        mem['param'] = module.weight.size().numel()

    elif isinstance(module, (nn.Sigmoid,)):
        ops['add'] = size_out.numel()
        ops['div'] = size_out.numel()
        ops['exp'] = size_out.numel()

        mem['act'] = size_out.numel()

    elif isinstance(module, (nn.LogSigmoid,)):
        ops['add'] = size_out.numel()
        ops['exp'] = 2 * size_out.numel()

        mem['act'] = size_out.numel()

    elif isinstance(module, (nn.Tanh,)):
        ops['add'] = 2 * size_out.numel()
        ops['div'] = size_out.numel()
        ops['exp'] = 2 * size_out.numel()

        mem['act'] = size_out.numel()

    elif isinstance(module, (nn.Softmin, nn.Softmax, nn.Softmax2d)):
        ops['add'] = size_out.numel()
        ops['div'] = size_out.numel()
        ops['exp'] = size_out.numel()

        mem['act'] = size_out.numel()

    elif isinstance(module, (nn.LogSoftmax,)):
        ops['add'] = 2 * size_out.numel()
        ops['exp'] = 2 * size_out.numel()

        mem['act'] = size_out.numel()

    else:
        pass

    d = {
        'name': names[modules.index(module)],
        'type': type(module).__name__,
        'batch': size_out[0],
        'ch_in': size_in[1],
        'ch_out': size_out[1],
        'dim_in': dims(size_in[2:]),
        'dim_out': dims(size_out[2:]),
        'ops': ops,
        'mem': mem,
        'desc': repr(module)
    }
    layers.append(d)


def reg_hook(module):
    module.register_forward_hook(hook)


# construct input dim for ptflops
def input_constructor(input_res, no_cuda=False):
    """
    :param input_res: tuple
    :return:input
    """
    if isinstance(input_res, tuple):
        input_size = [i for i in input_res]

    # batch_size of 2 for batchnorm
    if not no_cuda:
        dtype = torch.cuda.FloatTensor
    input = [torch.rand(1, *in_size).type(dtype) for in_size in input_size]
    return input


def main(net, x, summary):
    d = {
        'name': 'name',
        'type': 'type',
        'batch': 'batch',
        'ch_in': 'ch_in',
        'ch_out': 'ch_out',
        'dim_in': 'dim_in',
        'dim_out': 'dim_out',
        'ops': {
            'macc': 'macc',
            'cmp': 'cmp',
            'add': 'add',
            'div': 'div',
            'exp': 'exp'
        },
        'mem': {
            'act': 'act',
            'param': 'param',
            'buf': 'buf'
        },
        'desc': 'desc'
    }
    layers.append(d)

    for name, module in net.named_modules():
        names.append(name)
        modules.append(module)

    net.apply(reg_hook)

    with torch.no_grad():
        net(*x)

    ops = {
        'macc': 0,
        'cmp': 0,
        'add': 0,
        'div': 0,
        'exp': 0
    }
    mem = {
        'act': 0,
        'param': 0,
        'buf': 0
    }

    for layer in layers:
        if isinstance(layer['ops']['macc'], str):
            continue

        ops['macc'] += layer['ops']['macc']
        ops['cmp'] += layer['ops']['cmp']
        ops['add'] += layer['ops']['add']
        ops['div'] += layer['ops']['div']
        ops['exp'] += layer['ops']['exp']

        mem['act'] += layer['mem']['act']
        mem['param'] += layer['mem']['param']
        mem['buf'] += layer['mem']['buf']

    d = {
        'name': names[modules.index(net)],
        'type': type(net).__name__,
        'batch': layers[1]['batch'],
        'ch_in': layers[1]['ch_in'],
        'ch_out': layers[-1]['ch_out'],
        'dim_in': layers[1]['dim_in'],
        'dim_out': layers[-1]['dim_out'],
        'ops': ops,
        'mem': mem,
        'desc': ''  # repr(net)
    }
    layers.append(d)

    pre = ['%-', '%-', '%', '%', '%-', '%', '%-',
           '%', '%', '%', '%', '%',
           '%', '%', '%',
           '%-']
    lens = [0] * 16
    for i in range(len(layers)):
        d = layers[i]
        d = [d['name'], d['type'], d['batch'], d['ch_in'], d['dim_in'], d['ch_out'], d['dim_out'],
             d['ops']['macc'], d['ops']['cmp'], d['ops']['add'], d['ops']['div'], d['ops']['exp'],
             d['mem']['act'], d['mem']['param'], d['mem']['buf'],
             d['desc']]
        if not isinstance(d[7], str):
            for j in [2, 3, 5]:
                d[j] = str(d[j])
            for j in range(7, 15):
                d[j] = format(d[j], ',')
        layers[i] = d
        for j in range(len(lens)):
            if lens[j] < len('%s' % (d[j])):
                lens[j] = len('%s' % (d[j]))
    lens = [str(l) for l in lens]

    s = []
    for fmt in zip(pre, lens):
        s.append(''.join(fmt) + 's')
    s = ' '.join(s)

    if summary:
        print(s % tuple(layers[0]))
        print(s % tuple(layers[-1]))
    else:
        for d in layers:
            print(s % tuple(d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch nn analyzer')
    parser.add_argument('--path', type=str, default='./model.py',
                        help='python file containing network class definition')
    parser.add_argument('--name', type=str, default='Net',
                        help='class name of the network')
    parser.add_argument('--size', type=str, default='1,3,224,224',
                        help='size of input, comma separated')
    parser.add_argument('--summary', action='store_true',
                        help='only output summary of network')
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable CUDA, use CPU for forward')
    args = parser.parse_args()

    # path, filename = os.path.split(args.path)
    # sys.path.insert(0, path)
    # filename = os.path.splitext(filename)[0]
    # exec('from %s import %s as Net' % (filename, args.name))
    # size = [int(i) for i in args.size.split(',')]

    net = ComNet1(CE_dim=128, SD_n=[384, 120, 48])
    input_size = [(128,), (128,)]
    #x = [torch.rand(2, *in_size) for in_size in input_size]
    #x = torch.randn(*size)
    # if device == "cuda" and torch.cuda.is_available():
    #     dtype = torch.cuda.FloatTensor
    # else:
    #     dtype = torch.FloatTensor
    if not args.no_cuda:
        net = net.cuda()
        args.device = torch.device('cuda:0')
        dtype = torch.cuda.FloatTensor

    print("Complexity analysis with ptflops tool:...")
    print("###Analyze Comnet1###")
    macs, params = get_model_complexity_info(net, ((128,), (128,)), as_strings=True,
                                             input_constructor=partial(input_constructor),
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    net1 = ComNet2(CE_dim=128, SD_n=[128, 120, 48])
    net1 = net1.cuda()
    print("###Analyze Comnet2###")
    macs, params = get_model_complexity_info(net1, ((128,), (128,)), as_strings=True,
                                             input_constructor=partial(input_constructor),
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    net2 = ComNet3(CE_dim=128, hidden_size=[20, 10, 6], device=args.device)
    net2 = net2.cuda()
    print("###Analyze Comnet3###")
    macs, params = get_model_complexity_info(net2, ((128,), (128,)), as_strings=True,
                                             input_constructor=partial(input_constructor),
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print("###Analyze FC_Net###")
    net = FC_Net(256, 16)
    macs, params = get_model_complexity_info(net, (256,), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    x = [torch.rand(1, *in_size).type(dtype) for in_size in input_size]

    net.eval()

    with torch.no_grad():
        net(*x)

    #x = torch.randn(*size)

    prev_time = time.time()

    # if not args.no_cuda:
    #     x = x.cuda(non_blocking=True)

    with torch.no_grad():
        y = net(*x)

    if isinstance(y, torch.Tensor):
        y = y.cpu()
    else:
        y = y[0].cpu()

    print('Time cost: %.4f ms\n' % ((time.time() - prev_time) * 1000))

    main(net, x, args.summary)


