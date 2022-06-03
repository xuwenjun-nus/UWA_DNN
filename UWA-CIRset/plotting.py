#plotting logs
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def plot_curve(log, r, color, is_train):
    f = open(log, 'r')

    data = {}
    for ln in f:
        m = r.match(ln)
        if m is None:
            continue

        epoch = int(m.groups()[0])
        loss = float(m.groups()[1])
        try:
            data[epoch].append(loss)
        except:
            data[epoch] = [loss]

    f.close()

    losses = [[i, sum(data[i]) / len(data[i])] for i in data]

    x, y = np.array(losses).T
    label = log.replace(".log", " train" if is_train else " val")
    plt.plot(x, y, linestyle='--' if is_train else '-',
             color=color, linewidth=2, label=label)


def get_tick(lim):
    lg = np.log10(lim)
    lgexp = 10 ** np.floor(lg)
    frac = lim / lgexp
    if frac >= 8:
        return 2 * lgexp, 5 * lgexp / 10
    if frac >= 4:
        return 1 * lgexp, 2 * lgexp / 10
    elif frac >= 1.5:
        return 5 * lgexp / 10, 1 * lgexp / 10
    else:
        return 2 * lgexp / 10, 5 * lgexp / 100


def main():
    xlim = 200
    xmajor, xminor = get_tick(xlim)
    ylim = 0.3
    ymajor, yminor = get_tick(ylim)
    yprec = -np.floor(np.log10(yminor))

    plt.figure(figsize=(16, 9))
    ax = plt.subplot()
    ax.xaxis.set_major_locator(MultipleLocator(xmajor))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(xminor))
    ax.xaxis.set_minor_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(ymajor))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%%.%df' % yprec))
    ax.yaxis.set_minor_locator(MultipleLocator(yminor))
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%%.%df' % yprec))
    ax.grid(True, which='major', linewidth=1.5)
    ax.grid(True, which='minor', linewidth=0.5)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_xlim([0, xlim])
    ax.set_ylim([0, ylim])
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    for log, color in zip(args.logs, colors):
        plot_curve(log, res[0], color, is_train=True)
        plot_curve(log, res[1], color, is_train=False)
    plt.legend(loc='best')
    plt.savefig(args.out)


if __name__ == '__main__':
    res = [re.compile('.*Epoch\[ *(\d+)\] .*Training:  loss= *([-.\d]+).*'),
           re.compile('.*Epoch\[ *(\d+)\] Validation:  loss= *([-.\d]+).*')]

    parser = argparse.ArgumentParser(description='Plot training curve')
    parser.add_argument('--logs', nargs='+', type=str, default=None,
                        help='path to log file, space separated')
    parser.add_argument('--out', type=str, default='training-curve_FC.png',
                        help='name of output plot')
    args = parser.parse_args()
    main()
