import os
import sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']="0"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
import models.ood_detect as ood_detect
import os
import argparse
from models.resnet import ResNet50
from utils.train_utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--id', default=None, type=str, required = True,
                    help='In dataset')
parser.add_argument('--M',type=int,help='No of annotators',default=5)
parser.add_argument('--K',type=int,help='No of classes',default=10)
parser.add_argument('--model_arch', default='resnet50', type=str, help='model architecture ([densenet, resnet50,resnet34,resnet18]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='Epoch')
parser.add_argument('--bs', '--batch_size', default=128, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--seed', default=48, type=int,
                    help='Random Seed')
parser.add_argument('--name', default='ResNet50_cifar', type=str,
                    help='name of experiment', required = False)
parser.add_argument('--r', default=None, type=float, help='relevance ratio (0,1]', required = True)

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
in_dataset = args.id
kwargs = {'num_workers': 8, 'pin_memory': True}
if in_dataset == "CIFAR-10":
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x/255.0 for x in [63.0, 62.1, 66.7]])

elif in_dataset == "CIFAR-100":
    normalize = transforms.Normalize(mean=[0.507,0.487,0.441],
                                    std=[0.267,0.256,0.276])
else:
    raise Exception("Wrong Dataset")

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
    ])

if in_dataset == "CIFAR-10":
    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                        transform=transform_train),
        batch_size=args.bs, shuffle=True, **kwargs)
    valloader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform_test),
        batch_size=args.bs, shuffle=True, **kwargs)
    num_classes = 10

elif in_dataset == "CIFAR-100":
    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/nobackup/soumya/cifar100/data', train=True, download=True,
                        transform=transform_train),
        batch_size=args.bs, shuffle=True, **kwargs)
    valloader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/nobackup/soumya/cifar100/data', train=False, transform=transform_test),
        batch_size=args.bs, shuffle=True, **kwargs)
    num_classes = 100
else:
    raise Exception("Wrong Dataset")
print(f"Loading {args.id} with num classes = {num_classes}")

# Model
print('==> Building model..')

net = ood_detect.OOD_Detection( args.M, args.K, 50, args.model_arch)
net = net.to(device)
cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss().cuda()
loss_func_ce = torch.nn.NLLLoss(ignore_index=-1, reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)
if args.epoch == 100:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75,90], gamma = 0.1, verbose = True)
elif args.epoch == 200:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma = 0.1, verbose = True)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    lambda_reg =0.001
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        #outputs = net(inputs)
        # Compute DenseNet output
        final_output,B_weighted,h_softmax,reg_loss = net(inputs)  # Expected shape: [batch_size, M, K]
        #loss = criterion(outputs, targets)
        ce_loss =  loss_func_ce(final_output.log(), targets.long())	
        loss = ce_loss +lambda_reg*reg_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = final_output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(args, epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    lambda_reg =0.001
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            #outputs = net(inputs)
            final_output,B_weighted,h_softmax,reg_loss = net(inputs)  # Expected shape: [batch_size, M, K]
            #loss = criterion(outputs, targets)
            ce_loss =  loss_func_ce(final_output.log(), targets.long())	
            loss = ce_loss +lambda_reg*reg_loss
            #loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = final_output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # directory = "/nobackup/soumya/results/dice/%s/"%(args.name)
        directory = os.path.join(f"./checkpoints/{args.id}/resnet50")
        if not os.path.exists(directory):
             os.makedirs(directory)
        filename = os.path.join(directory,'model_best.pth.tar')
        torch.save(state, filename)
        best_acc = acc
def adjust_learning_rate(optimizer, epoch, tot_epochs):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    if tot_epochs == 300:
         lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    elif tot_epochs == 200:
         lr = args.lr * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 75)) * (0.1 ** (epoch // 90))
    elif tot_epochs == 100:
         lr = args.lr * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 75)) * (0.1 ** (epoch // 90))
    else:
        raise Exception("Check Epochs")
    # log to TensorBoard
    print(f"Current lr: {lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
print(f"Training for {args.epoch} epochs")
for epoch in range(start_epoch, start_epoch + args.epoch):
    #adjust_learning_rate(optimizer, epoch, args.epoch)
    train(epoch)
    test(args, epoch)
    scheduler.step()
print(f"Best Accuracy :{best_acc}")
