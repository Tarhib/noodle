# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']="0"
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.densenet as dn
from torch.optim import Adam
#from NoiseLabelDataset import NoiseLabelDataset
import torch.nn.functional as F  # Add this line to import the functional module
import models.ood_detect as ood_detect
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.cifar import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset  # Fix: Import Dataset explicitly

# Confusion Matrix Class
'''
class sig_t(nn.Module):
    def __init__(self, device, num_classes, init=2):
        super(sig_t, self).__init__()
        self.register_parameter(name='w', param=nn.parameter.Parameter(-init * torch.ones(num_classes, num_classes)))
        self.w.to(device)

        co = torch.ones(num_classes, num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.to(device)
        self.identity = torch.eye(num_classes).to(device)

    def forward(self):
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig * self.co.detach()
        T = F.normalize(T, p=1, dim=1)
        return T
        
class SigT(nn.Module):
    def __init__(self, device, num_classes, num_matrices, init=2):
        super(SigT, self).__init__()
        # Create multiple confusion matrices with dimensions (num_matrices, num_classes, num_classes)
        self.register_parameter(
            name='w',
            param=nn.Parameter(-init * torch.ones(num_matrices, num_classes, num_classes))
        )
        self.w.to(device)

        # Create a mask to retain identity in each confusion matrix (num_matrices, num_classes, num_classes)
        co = torch.ones(num_classes, num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.to(device).expand(num_matrices, -1, -1)  # Broadcasting across matrices
        self.identity = torch.eye(num_classes).to(device).expand(num_matrices, -1, -1)

    def forward(self):
        # Apply sigmoid to learnable parameters for each matrix
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig * self.co.detach()
        # Normalize each matrix individually
        T = F.normalize(T, p=1, dim=2)
        return T
'''      

       
parser = argparse.ArgumentParser(description='PyTorch DenseNet Training with Confusion Matrix')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.1)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--M',type=int,help='No of annotators',default=6)
parser.add_argument('--K',type=int,help='No of classes',default=100)
parser.add_argument('--id', default=None, type=str, required=True, help='In dataset')
parser.add_argument('--temp', default=0.04, type=float, help='Temperature')
parser.add_argument('--seed', default=0, type=int, help='Random Seed')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--bs', '--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=50, type=int, help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int, help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int, help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=0.5, type=float, help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet-101_cifar100', type=str, help='name of experiment')
parser.add_argument('--r', default=None, type=float, help='relevance ratio', required=True)
parser.add_argument('--start_prune', default=40, type=int,
                    help='number of total epochs to run')
# ADD ResNet18 support argument
parser.add_argument('--model_type', default='densenet', type=str, choices=['densenet', 'resnet18'], help='model type: densenet or resnet18')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0
#selected_noise_label = 'rand3'  # Change this to aggre_label, random_label1, etc.
class GCELoss(nn.Module):
    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(GCELoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
        # Ensure weight is on same device as logits
        if self.weight.device != logits.device:
            self.weight.data = self.weight.data.to(logits.device)
        
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)
        return loss
    
    def update_weight(self, logits, targets, indexes):
        # Ensure weight is on same device as logits
        if self.weight.device != logits.device:
            self.weight.data = self.weight.data.to(logits.device)
            
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        
        # Create k term on same device as logits
        Lqk = torch.full((targets.size(0),), (1-(self.k**self.q))/self.q, device=logits.device)
        Lqk = torch.unsqueeze(Lqk, 1)
        
        condition = torch.gt(Lqk, Lq)
        
        # FIX: Update .data attribute instead of direct assignment
        self.weight.data[indexes] = condition.float()
        
        
        
        
# SCE Loss Implementation
class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
def get_loss_function(loss_type, num_classes=10, ignore_index=-1):
    """
    Factory function to get different loss functions with standard parameters from papers
    """
    if loss_type == 'gce':
        return GCELoss()  # Standard q=0.7 from paper
    elif loss_type == 'sce':
        return SCELoss(alpha=0.1, beta=1.0, num_classes=num_classes)  # Standard a=0.1, ß=1.0 from paper
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def main():
    global args, best_prec1
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device setup
    device = torch.device("cuda")

    # Load CIFAR-10 noisy labels
    # Data loading code
    #CIFAR10
    # Load CIFAR-10 noisy labels
    # Load noise files for both datasets
    if args.id == "CIFAR-10":
        noise_file = torch.load('./data/CIFAR-10_human.pt')
        clean_label   = noise_file['clean_label']
        worst_label   = noise_file['worse_label'] 
        aggre_label   = noise_file['aggre_label']
        random_label1 = noise_file['random_label1']
        random_label2 = noise_file['random_label2']
        random_label3 = noise_file['random_label3']
        
        # Choose a noisy label version (modify as needed)
        selected_noise_label = worst_label  # Change this to aggre_label, random_label1, etc.
        selected_noise_name = "worse_label"  # Keep track of the name for logging
        print(selected_noise_name)
        
    elif args.id == "CIFAR-100":
        noise_file = torch.load('./data/CIFAR-100_human.pt')
        clean_label = noise_file['clean_label']
        noise_label = noise_file['noisy_label']  # or 'noise_label' depending on file
        
        # CIFAR-100N only has one noise type
        selected_noise_label = noise_label  # Could be either coarse (25.60%) or fine (40.20%)
        selected_noise_name = "noisy_label"
        
        # To check which type you have, you can calculate noise rate:
        noise_rate = (torch.tensor(clean_label) != torch.tensor(noise_label)).float().mean()
        print(f"CIFAR-100N noise rate: {noise_rate:.2%}")
        if abs(noise_rate - 0.256) < 0.01:
            print("This appears to be CIFAR-100N Coarse")
            selected_noise_name = "coarse_noisy"
        elif abs(noise_rate - 0.402) < 0.01:
            print("This appears to be CIFAR-100N Fine") 
            selected_noise_name = "fine_noisy"
        print(selected_noise_name)
    
    # Custom Dataset classes to Apply Noisy Labels
    class CIFAR10Noisy(Dataset):
        def __init__(self, trainset, noise_labels):
            self.dataset = trainset
            self.noise_labels = noise_labels
            assert len(self.dataset) == len(self.noise_labels), "Mismatch between dataset and noisy labels!"
    
        def __getitem__(self, index):
            image, _ = self.dataset[index]  # Ignore original label
            noisy_target = self.noise_labels[index]  # Use noisy label
            return image, noisy_target
    
        def __len__(self):
            return len(self.dataset)
    
    class CIFAR100Noisy(Dataset):
        def __init__(self, trainset, noise_labels):
            self.dataset = trainset
            self.noise_labels = noise_labels
            assert len(self.dataset) == len(self.noise_labels), "Mismatch between dataset and noisy labels!"
    
        def __getitem__(self, index):
            image, _ = self.dataset[index]  # Ignore original label
            noisy_target = self.noise_labels[index]  # Use noisy label
            return image, noisy_target
    
        def __len__(self):
            return len(self.dataset)
            
            
    
    class CIFAR100Noisy_GCE(Dataset):
        def __init__(self, trainset, noise_labels):
            self.dataset = trainset
            self.noise_labels = noise_labels
            assert len(self.dataset) == len(self.noise_labels), "Mismatch between dataset and noisy labels!"
    
        def __getitem__(self, index):
            image, _ = self.dataset[index]  # Ignore original label
            noisy_target = self.noise_labels[index]  # Use noisy label
            return image, noisy_target, index  # ADD INDEX HERE
    
        def __len__(self):
            return len(self.dataset)

    # Data loading code
    in_dataset = args.id
    if in_dataset == "CIFAR-10":
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    elif in_dataset == "CIFAR-100":
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
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

    kwargs = {'num_workers': 8, 'pin_memory': True}

    if in_dataset == "CIFAR-10":
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        
        #trainset=datasets.CIFAR10('./data', train=True, download=True,
                            #transform=transform_train)
        trainset_NoiseDataset= CIFAR10Noisy(trainset, selected_noise_label)
        #trainset_NoiseDataset = NoiseLabelDataset(trainset, ErrorRate=0.4, ShowErrorLabel=True)
        #trainset_noisy = CIFAR10Noisy(trainset, selected_noise_label)
        train_loader = torch.utils.data.DataLoader(
            trainset_NoiseDataset,
            batch_size=args.bs, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transform_test),
            batch_size=args.bs, shuffle=False, **kwargs)
        num_classes = 10
    elif in_dataset == "CIFAR-100":
        
        num_classes=100
        
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainset_NoiseDataset= CIFAR100Noisy(trainset, selected_noise_label)
        train_loader = torch.utils.data.DataLoader(
            trainset_NoiseDataset,
            batch_size=args.bs, shuffle=True, **kwargs)
    
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transform_test),
            batch_size=args.bs, shuffle=False, **kwargs)
        num_classes = 100
    else:
        raise Exception("Wrong Dataset")

    print(f"Loading {args.id} with num classes = {num_classes}")
    
    # MODIFIED: Create model based on model_type argument
    if args.model_type == 'densenet':
        model = ood_detect.OOD_Detection(args.M, args.K, args.layers, 'densenet')
    elif args.model_type == 'resnet18':
        model = ood_detect.OOD_Detection(args.M, args.K, args.layers, 'resnet18', r=0.05)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    print(f'Using {args.model_type} model')

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    model = model.cuda()
    #penultimate_size= model.in_planes
    #trans = sig_t(device=device, num_classes=num_classes, penultimate_size=penultimate_size).to(device)  # Ensure confusion matrix is also on the same device

    #trans_matrices = ood_detect.SigT(device=device,M= args.M, K=args.K,).to(device)
    
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    # Loss function and optimizers
    #criterion = nn.CrossEntropyLoss().cuda()
    #loss_func_ce = torch.nn.NLLLoss(ignore_index=-1, reduction='mean')
    criterion = get_loss_function('sce', num_classes=num_classes, ignore_index=-1) 
    loss_func_ce = F.nll_loss
    optimizer = torch.optim.SGD(model.fnet.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
   
    #optimizer_trans = torch.optim.SGD(model.trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-5)
    optimizer_trans = torch.optim.Adam(model.trans.parameters(), lr=args.lr)
    #scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler_f = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    
    
    
   

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.epochs)
        adjust_learning_rate(optimizer_trans, epoch, args.epochs)

        # Train for one epoch
        
        train(train_loader, model, loss_func_ce,optimizer, epoch,num_classes,criterion,args,scheduler_f,optimizer_trans)
        #scheduler.step()
        # Evaluate on validation set
        prec1 = validate(val_loader, model, loss_func_ce,epoch,criterion,args)

        # Remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        #all_trans_matrices = trans_matrices()
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=f"checkpoint_{epoch}.pth.tar")

    print('Best accuracy: ', best_prec1)

# CLEANER SOLUTION: Replace your training loop unpacking with direct unpacking

def train(train_loader, model, loss_func_ce, optimizer, epoch, num_classes, criterion, args, scheduler_f, optimizer_trans):
    batch_time = AverageMeter()
    losses = AverageMeter()
    ce_loss_meter = AverageMeter()
    kl_loss_meter = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()
    total_samples = len(train_loader.dataset)
    batch_size = args.bs
    M = args.M
    K = args.K
    lambda_sparse = 0.0001

    # CHANGED: Direct unpacking of all 3 values from CIFAR100Noisy_GCE
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input = input.cuda()
        #indexes = indexes.cuda()  # Move indexes to GPU
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        optimizer_trans.zero_grad()

        # Compute DenseNet output
        final_output, reg_loss = model(input_var)
        
        # FIXED: Pass all 3 arguments to GCE loss
        ce_loss = criterion(final_output, target_var)
        loss = ce_loss + lambda_sparse * reg_loss
   
        prec1 = accuracy(final_output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_trans.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')
     
    print(f'Epoch: [{epoch}]\t'
          f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
          f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

def validate(val_loader, model, loss_func_ce, epoch, criterion, args):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    kl_loss_meter = AverageMeter()
    top1 = AverageMeter()
   
    batch_size = args.bs
    M = args.M
    K = args.K
    lambda_sparse = 0.0001
    
    model.eval()
    end = time.time()

    with torch.no_grad():
        # CHANGED: Standard validation dataset only provides 2 values
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            
            # Create dummy indexes for validation
            indexes = torch.arange(len(target)).cuda()
            
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            final_output, reg_loss = model(input_var)
            
            # Pass dummy indexes to GCE loss
            ce_loss = criterion(final_output, target_var)
            loss = ce_loss + lambda_sparse * reg_loss
            prec1 = accuracy(final_output.data, target, topk=(1,))[0]

            top1.update(prec1, input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


# The rest of the code (save_checkpoint, adjust_learning_rate, accuracy, etc.) remains the same.
def save_checkpoint(args, state, is_best, filename):
    """Saves checkpoint to disk"""
    # MODIFIED: Include model_type in checkpoint directory
    if args.model_type == 'densenet':
        directory = os.path.join("./checkpoints", f"{args.id}", "densenet_noodle_sce_combo_bs_128")
    elif args.model_type == 'resnet18':
        directory = os.path.join("./checkpoints", f"{args.id}", "resnet18_cm_rank_20_std")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory,filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory,'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()