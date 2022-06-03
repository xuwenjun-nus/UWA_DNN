from model import FC_Net, ComNet1, CENet, count_parameters
from data_generator import UWAComDataset
import time
from config import argument
import logging
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch import cuda
from torch.utils import data
from torch.backends import cudnn
from utils import *
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


def log_gradients_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            logger.add_histogram(tag + "/grad", value.grad.cpu(), step)


def initialize_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        nn.init.constant_(layer.bias, 0.0)


def initialize_weights_SD(layer, W):
    if type(layer) == nn.Linear:
        with torch.no_grad():
            layer.weight = W


def train(args, net, trainloader, criterion, optimizer, epoch, tb):
    net.train()

    running_loss = 0.0
    total_loss = 0.0
    prev_time = time.time()
    step = 0 # NO. of training steps
    for i, (y_d, h_ls, true_H, bits) in enumerate(trainloader):
        step += 1
        if args.model == 'CENet':
            input = h_ls
            label = true_H

        input = input.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)

        output = net(input)
        output = output.view(-1, output.size()[-1])
        label = label.view(-1, label.size()[-1])

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        # train for 50 minibatches
        if (i + 1) == 50:
            break

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch[%4d] Iter[%4d] Training:  loss=%6.4f    speed=%8.2f sample/sec',
                        epoch + 1, i + 1,
                        running_loss / args.print_freq, # loss every iter
                        args.batch_size * args.print_freq / (time.time() - prev_time))
            running_loss = 0.0
            prev_time = time.time()
        #add logging info for tensorboard
        # add logging info for tensorboard
        tb.add_scalar("Running Loss/Batches", total_loss / step, epoch * args.minibatch_size + step)
        # tb.add_scalar("BER", total_correct / len(train_set), epoch)
        log_gradients_in_model(net, tb, epoch * args.minibatch_size + step)
    tb.add_scalar("Loss/Epochs", total_loss / args.minibatch_size, epoch)


def val(args, net, valloader, criterion, epoch, tb):
    net.eval()

    total_loss = 0.0
    prev_time = time.time()

    with torch.no_grad():
        for _,  (y_d, h_ls, true_H, bits) in enumerate(valloader):
            if args.model == 'CENet':
                input = h_ls
                gt = true_H
            else:
                input = (y_d, h_ls)
                gt = bits
            input = input.to(args.device, non_blocking=True)
            gt = gt.to(args.device, non_blocking=True)

            output = net(input)
            # the next two lines are for KL divergence
            output = output.view(-1, output.size()[-1])
            gt = gt.view(-1, gt.size()[-1])

            loss = criterion(output, gt)

            total_loss += loss.item() * input.size()[0]

    logger.info('Epoch[%4d] Validation:  loss=%6.4f    speed=%8.2f sample/sec',
                epoch + 1,
                total_loss / len(valloader.dataset),
                len(valloader.dataset) / (time.time() - prev_time))
    tb.add_scalar("Evaluation Loss/Epochs", total_loss / len(valloader.dataset), epoch)
    return total_loss / len(valloader.dataset)


def comnet_train_step(args, net, trainloader, criterion, optimizer, epoch, tb):
    net.train()
    running_loss = 0.0
    total_loss = 0.0
    prev_time = time.time()
    step = 0  # NO. of training steps
    for i, (input1, input2, input3, label) in enumerate(trainloader):
        step += 1
        y_d = input1.to(args.device, non_blocking=True)
        h_ls = input2.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)

        output = net(y_d, h_ls)
        # the next two lines are for KL divergence
        output = output.view(-1, output.size()[-1])
        label = label.view(-1, label.size()[-1])

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        # train for 50 minibatches
        if (i+1) == 50:
            break

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch[%4d] Iter[%4d] Training:  loss=%6.4f    speed=%8.2f sample/sec',
                        epoch + 1, i + 1,
                        running_loss / args.print_freq,  # loss every iter
                        args.batch_size * args.print_freq / (time.time() - prev_time))
            running_loss = 0.0
            prev_time = time.time()
        # add logging info for tensorboard
        tb.add_scalar("Running Loss/Batches", total_loss / step, epoch * args.minibatch_size + step)
        # tb.add_scalar("BER", total_correct / len(train_set), epoch)
        log_gradients_in_model(net, tb, epoch * args.minibatch_size + step)
    tb.add_scalar("Loss/Epochs", total_loss / args.minibatch_size, epoch)


def comnet_val_step(args, net, valloader, criterion, epoch, tb):
    net.eval()
    total_loss = 0.0
    prev_time = time.time()
    with torch.no_grad():

        for i, (input1, input2, input3, label) in enumerate(valloader):

            y_d = input1.to(args.device, non_blocking=True)
            h_ls = input2.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)

            output = net(y_d, h_ls)
            # the next two lines are for KL divergence
            output = output.view(-1, output.size()[-1])
            label = label.view(-1, label.size()[-1])

            loss = criterion(output, label)  # loss per-item
            total_loss += loss.item()*input1.size()[0]

    logger.info('Epoch[%4d] Validation:  loss=%6.4f    speed=%8.2f sample/sec',
                epoch + 1,
                total_loss / len(valloader.dataset),
                len(valloader.dataset) / (time.time() - prev_time))

    tb.add_scalar("Evaluation Loss/Epochs", total_loss / len(valloader.dataset), epoch)
    return total_loss / len(valloader.dataset)


def main():
    args = argument()
    logger.info(args)
    cudnn.benchmark = True
    modem = QAMModem(K)
    trainset = UWAComDataset(CIR_dir=Train_set_path, idx_low=1, idx_high=300, modem=modem)
    #trainset = UWADNNDataset(CIR_dir=Train_set_path, idx_low=1, idx_high=300)
    trainloader = data.DataLoader(trainset, args.batch_size, shuffle=True,
                                  num_workers=args.num_worker, pin_memory=True)
    valset= UWAComDataset(CIR_dir=Train_set_path, idx_low=301, idx_high=400, modem=modem)
    logger.info('Training set: %6d samples, %4d batches',
                len(trainset), len(trainloader))
    #valset = UWADNNDataset(CIR_dir=Train_set_path, idx_low=301, idx_high=400)
    valloader = data.DataLoader(valset, args.batch_size, shuffle=False,
                                num_workers=args.num_worker, pin_memory=True)
    logger.info('Validation set: %6d samples %4d batches',
                len(valset), len(valloader))
    writer = SummaryWriter(LOG_DIR)
    #net = FC_Net(256, 16)
    #net = ComNet1(CE_dim=128, SD_n=[384, 120, 48])
    # perform LMMSE estimation first and obtain W
    W = torch.as_tensor(weight_init(), dtype=torch.float)
    net = CENet(CE_dim=128, W=W)
    fake_input = torch.randn(1, 128)
    writer.add_graph(net, fake_input)
    args.no_cuda = args.no_cuda or not cuda.is_available()
    if not args.no_cuda:
        args.device = torch.device('cuda:0')
        net = nn.DataParallel(net).to(args.device)
        logger.info('Using GPU: all')
    else:
        args.device = torch.device('cpu')
        net = net.to(args.device)
        logger.info('Using CPU')

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1)

    best_loss = 1000000
    print('Training start')
    CE_epoch = 2000
    for epoch in range(args.start_epoch, CE_epoch):
        logger.info('Epoch[%4d] Training:  lr=%.1e',
                    epoch + 1, optimizer.param_groups[0]['lr'])

        train(args, net, trainloader, criterion, optimizer, epoch, writer)

        # if (epoch + 1) % args.save_freq == 0:
        #     path = os.path.join(args.save_path,
        #                         'model_%d_%s.pth' % (epoch + 1, time.strftime('%y-%m-%d-%H-%M-%S', time.localtime())))
        #     torch.save(net.module.state_dict(), path)
        #     logger.info('Epoch[%4d] Training:  saving checkpoint to %s',
        #                 epoch + 1, path)
        #     torch.save(optimizer.state_dict(),
        #                os.path.join(args.save_path, 'optimizer_%d.pth' % (epoch + 1)))

        if (epoch + 1) % args.val_freq == 0:
            loss = val(args, net, valloader, criterion, epoch, writer)

            if loss < best_loss:
                for f in os.listdir(args.save_path):
                    if f[:8] == 'CE_best_':
                        os.remove(os.path.join(args.save_path, f))
                path = os.path.join(args.save_path,
                                    'CE_best_%6.4f_%d.pth' % (loss, epoch + 1))
                torch.save(net.module.state_dict(), path)
                logger.info('Epoch[%4d] Validation:  saving best model to %s',
                            epoch + 1, path)
                torch.save(optimizer.state_dict(),
                           os.path.join(args.save_path, 'CE_best_optimizer_%d.pth' % (epoch + 1)))
                best_loss = loss  # val loss
        scheduler.step()

    print('Training finish')
    print('best loss:', best_loss)


def sequential_training(resume=False):
    "sequential training for Comnet"
    args = argument()
    logger.info(args)
    cudnn.benchmark = True
    # perform LMMSE estimation first and obtain W
    #W = torch.as_tensor(weight_init(), dtype=torch.float)
    net = ComNet1(CE_dim=128, SD_n=[384, 120, 48])
    writer = SummaryWriter(LOG_DIR)
    # net = FC_Net(256, 16)
    # list of model summary
    fake_input = torch.randn(1, 128)
    # y, h, label = next(iter(trainloader))
    # predict = net(y, h)
    writer.add_graph(net, [fake_input, fake_input])
    modem = QAMModem(K)
    args.no_cuda = args.no_cuda or not cuda.is_available()
    if not args.no_cuda:
        args.device = torch.device('cuda:0')
        net = nn.DataParallel(net).to(args.device)
        logger.info('Using GPU: all')
    else:
        args.device = torch.device('cpu')
        net = net.to(args.device)
        logger.info('Using CPU')
    # count net parameters
    # print('The total number of parameters in the model is: %8d!' % count_parameters(net))
    # summary(net, [(128,), (128,)])
    trainset = UWAComDataset(CIR_dir=Train_set_path, idx_low=1, idx_high=300, modem=modem)
    trainloader = data.DataLoader(trainset, args.batch_size, shuffle=True,
                                     num_workers=args.num_worker, pin_memory=True)
    valset = UWAComDataset(CIR_dir=Train_set_path, idx_low=301, idx_high=400, modem=modem)
    logger.info('Training set: %6d samples, %4d batches',
                   len(trainset), len(trainloader))

    valloader = data.DataLoader(valset, args.batch_size, shuffle=False,
                                   num_workers=args.num_worker, pin_memory=True)
    logger.info('Validation set: %6d samples %4d batches',
                   len(valset), len(valloader))
    CE_epoch = 2000
    SD_eposh = 5000
    if not resume:
        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1)
        best_loss = 1000000
        print('Training start')

    # train CE first for CE epochs
        for epoch in range(args.start_epoch, CE_epoch):
            logger.info('Epoch[%4d] Training:  lr=%.1e',
                        epoch + 1, optimizer.param_groups[0]['lr'])

            comnet_train_step(args, net, trainloader, criterion, optimizer, epoch, writer)

            if (epoch + 1) % args.val_freq == 0:
                loss = comnet_val_step(args, net, valloader, criterion, epoch, writer)
                if loss < best_loss:

                    for f in os.listdir(args.save_path):
                        if f[:8] == 'best_':
                            os.remove(os.path.join(args.save_path, f))
                    path = os.path.join(args.save_path,
                                           'CE_best_%6.4f_%d.pth' % (loss, epoch + 1))
                    torch.save(net.module.state_dict(), path)
                    logger.info('Epoch[%4d] Validation:  saving best model to %s',
                                   epoch + 1, path)
                    torch.save(optimizer.state_dict(),
                                os.path.join(args.save_path, 'CE_best_optimizer_%d.pth' % (epoch + 1)))
                    best_loss = loss  # val loss
            scheduler.step()
        print('best loss:', best_loss)
    else:
        #load trained checkpoint model
        best_loss = 100
        path = os.path.join(args.save_path, 'CE_best_0.0064_1800.pth')
        print(path)
        if isinstance(net, nn.DataParallel):
            net.module.load_state_dict(torch.load(path), strict=False)
        # fix CE parameters and train SD only
        for param in net.module.CE.parameters():
            param.requires_grad = False
        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.module.parameters()), lr=args.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 3000, 4000], gamma=0.5)
        for epoch in range(SD_eposh):
            logger.info('Epoch[%4d] Training:  lr=%.1e',
                        epoch + 1, optimizer.param_groups[0]['lr'])

            comnet_train_step(args, net, trainloader, criterion, optimizer, epoch, writer)

            if (epoch + 1) % args.val_freq == 0:
                loss = comnet_val_step(args, net, valloader, criterion, epoch, writer)

                if loss < best_loss:
                    for f in os.listdir(args.save_path):
                        if f[:8] == 'SD_best_':
                            os.remove(os.path.join(args.save_path, f))
                    path = os.path.join(args.save_path,
                                        'SD_best_%6.4f_%d.pth' % (loss, epoch + 1))
                    torch.save(net.module.state_dict(), path)
                    logger.info('Epoch[%4d] Validation:  saving best model to %s',
                                epoch + 1, path)
                    torch.save(optimizer.state_dict(),
                               os.path.join(args.save_path, 'SD_best_optimizer_%d.pth' % (epoch + 1)))
                    best_loss = loss  # val loss
            scheduler.step()

        print('Training finish')
        print('best loss:', best_loss)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(name)s:%(levelname)s:  %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    fh = logging.FileHandler(
        'train_' + time.strftime('%y-%m-%d-%H-%M-%S', time.localtime()) + '.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    time_start = time.time()

    #main()
    sequential_training(resume=True)
    time_elapse = time.time() - time_start
    print("time elapsed: " + str(time_elapse))
