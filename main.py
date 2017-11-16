'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import math
import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable


try:
    from bashplotlib.scatterplot import plot_scatter

    def scatter(yvals):
        import numpy as np
        
        xvals = np.arange(len(yvals))
        yvals = np.array(yvals)
        csv = np.hstack([xvals[:,np.newaxis], yvals[:,np.newaxis]])

        np.savetxt('bashplot.txt', csv, delimiter=',')
        
        plot_scatter('bashplot.txt', None, None, 40, 'o', 'default', 'Learning Rate Schedule')

        os.remove('bashplot.txt')
    can_plot = True
except ImportError:
    can_plot = False

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
parser.add_argument('--lr_period', default=10, type=float, help='learning rate schedule restart period')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--plot', '-p', action='store_true', help='plot the learning rate')
parser.add_argument('--save', '-s', action='store_true', help='saves state_dict on every epoch (for resuming best performing model and saving it)')
parser.add_argument('--sparsify', action='store_true', help='sparsify on warm restarts')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

lr_period = args.lr_period*len(trainloader)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet50()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True



if args.sparsify:
    from deep_compression import MaskedSGD
    SGD = MaskedSGD
else:
    SGD = optim.SGD
    
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def sgdr(period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx/restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(batch_idx/restart_period)
    return 0.5*(1.0 + math.cos(radians))

lr_trace = []
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    global optimizer
    start_batch_idx = len(trainloader)*epoch
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        global_step = batch_idx+start_batch_idx
        batch_lr = args.lr*sgdr(lr_period, global_step)
        lr_trace.append(batch_lr)
        optimizer = set_optimizer_lr(optimizer, batch_lr)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if len(lr_trace) > 1:
            if lr_trace[-1] - lr_trace[-2] > 1e-3:
                # we've just reset the learning rate
                if args.sparsify:
                    print("Sparsifying at step %i..."%global_step)
                    optimizer.sparsify()
                print("Sparsitying is %f"%optimizer.sparsity())
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | LR: %.3f'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, batch_lr))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    if can_plot and args.plot:
        scatter(lr_trace)
    if args.save:
        torch.save(net.module.state_dict(), 'checkpoint/state_dict.pth')
