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
import torch.nn.functional as F
import models.ood_detect as ood_detect
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.cifar import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset


parser = argparse.ArgumentParser(description='PyTorch DenseNet Training with Confusion Matrix')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.5)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--M',type=int,help='No of annotators',default=6)
parser.add_argument('--K',type=int,help='No of classes',default=10)
parser.add_argument('--id', default=None, type=str, required=True, help='Dataset selection: [CIFAR-10, CIFAR-100, Animal10n]')
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
parser.add_argument('--start_prune', default=40, type=int, help='number of total epochs to run')
parser.add_argument('--model_type', default='densenet', type=str, choices=['densenet', 'resnet18'], help='model type: densenet or resnet18')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0

class GCELoss(nn.Module):
    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(GCELoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
        if self.weight.device != logits.device:
            self.weight.data = self.weight.data.to(logits.device)
        
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)
        return loss
    
    def update_weight(self, logits, targets, indexes):
        if self.weight.device != logits.device:
            self.weight.data = self.weight.data.to(logits.device)
            
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        
        Lqk = torch.full((targets.size(0),), (1-(self.k**self.q))/self.q, device=logits.device)
        Lqk = torch.unsqueeze(Lqk, 1)
        
        condition = torch.gt(Lqk, Lq)
        self.weight.data[indexes] = condition.float()

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

def get_loss_function(loss_type, num_classes=10, ignore_index=-1, trainset_size=50000):
    if loss_type == 'gce':
        return GCELoss(trainset_size=trainset_size)
    elif loss_type == 'sce':
        return SCELoss(alpha=0.1, beta=1.0, num_classes=num_classes)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Custom wrapper for Animal10n to include indexes (Fixed for Deep Lake 3.x API)
class Animal10nWrapper(Dataset):
    def __init__(self, deeplake_dataset, transform_func, use_cache=False, return_index=True):
        import time  # Import for timing
        self.deeplake_dataset = deeplake_dataset  # ds_train IS the dataset
        self.transform_func = transform_func
        self.dataset_size = len(deeplake_dataset)  # Fixed: Direct length access
        self.use_cache = use_cache
        self.return_index = return_index  # Whether to return index (for training) or not (for validation)
        
        if use_cache:
            # Pre-load and cache the data for consistent indexing
            self.cached_data = []
            print(f"Caching Animal10n data for consistent indexing... (this will take ~2-3 minutes)")
            start_time = time.time()
            for i, sample in enumerate(deeplake_dataset):  # Fixed: Direct iteration
                self.cached_data.append(sample)
                if i % 500 == 0:  # More frequent updates
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (self.dataset_size - i - 1) / rate if rate > 0 else 0
                    print(f"Cached {i+1}/{self.dataset_size} samples ({rate:.1f} samples/sec, ETA: {eta/60:.1f} min)")
            print(f"Caching complete! Total time: {(time.time() - start_time)/60:.1f} minutes")
        else:
            print("Using direct access mode (no caching) - will be VERY slow for training!")
    
    def __getitem__(self, index):
        if self.use_cache:
            sample = self.cached_data[index]
        else:
            sample = self.deeplake_dataset[index]  # Fixed: Direct indexing
            
        transformed = self.transform_func(sample)
        
        # Handle tensor conversion for labels
        image = transformed['images']
        label = transformed['labels']
        
        # Convert Deep Lake label to Python int
        if hasattr(label, 'numpy'):
            label = label.numpy()
        if isinstance(label, np.ndarray):
            if label.ndim > 0:
                label = label.item()  # Convert to scalar
        elif isinstance(label, torch.Tensor):
            if label.dim() > 0:
                label = label.squeeze()
            if label.dim() == 0:
                label = label.item()
        
        # Ensure label is integer
        if not isinstance(label, int):
            label = int(label)
        
        # Return with or without index based on use case
        if self.return_index:
            return image, label, index  # For training (GCE loss needs indexes)
        else:
            return image, label  # For validation (standard format)
    
    def __len__(self):
        return self.dataset_size

def main():
    global args, best_prec1
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device setup
    device = torch.device("cuda")

    # Load noise files for datasets
    if args.id == "CIFAR-10":
        noise_file = torch.load('./data/CIFAR-10_human.pt')
        clean_label   = noise_file['clean_label']
        worst_label   = noise_file['worse_label'] 
        aggre_label   = noise_file['aggre_label']
        random_label1 = noise_file['random_label1']
        random_label2 = noise_file['random_label2']
        random_label3 = noise_file['random_label3']
        
        selected_noise_label = random_label2
        selected_noise_name = "random_label2"
        print(selected_noise_name)
        
    elif args.id == "CIFAR-100":
        noise_file = torch.load('./data/CIFAR-100_human.pt')
        clean_label = noise_file['clean_label']
        noise_label = noise_file['noisy_label']
        
        selected_noise_label = noise_label
        selected_noise_name = "noisy_label"
        
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
            image, _ = self.dataset[index]
            noisy_target = self.noise_labels[index]
            return image, noisy_target, index  # Added index for GCE loss
    
        def __len__(self):
            return len(self.dataset)
    
    class CIFAR100Noisy(Dataset):
        def __init__(self, trainset, noise_labels):
            self.dataset = trainset
            self.noise_labels = noise_labels
            assert len(self.dataset) == len(self.noise_labels), "Mismatch between dataset and noisy labels!"
    
        def __getitem__(self, index):
            image, _ = self.dataset[index]
            noisy_target = self.noise_labels[index]
            return image, noisy_target, index  # Added index for GCE loss
    
        def __len__(self):
            return len(self.dataset)

    # Data loading code
    in_dataset = args.id
    
    if in_dataset == "CIFAR-10":
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        num_classes = 10
    elif in_dataset == "CIFAR-100":
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        num_classes = 100
    elif in_dataset == "Animal10n":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        num_classes = 10
        print(f"Animal10n setup complete!")
        print("Training will start after caching completes...")
        print("TIP: Caching happens once at startup, then training is fast!")
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
        trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        trainset_NoiseDataset = CIFAR10Noisy(trainset, selected_noise_label)
        trainset_size = len(trainset)
        
        train_loader = torch.utils.data.DataLoader(
            trainset_NoiseDataset, batch_size=args.bs, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transform_test),
            batch_size=args.bs, shuffle=False, **kwargs)
            
    elif in_dataset == "CIFAR-100":
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainset_NoiseDataset = CIFAR100Noisy(trainset, selected_noise_label)
        trainset_size = len(trainset)
        
        train_loader = torch.utils.data.DataLoader(
            trainset_NoiseDataset, batch_size=args.bs, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transform_test),
            batch_size=args.bs, shuffle=False, **kwargs)
            
    elif in_dataset == "Animal10n":
        print("Loading Animal10n dataset using Deep Lake...")
        print("NOTE: First epoch will be slow due to caching (~3 min), subsequent epochs will be fast!")
        
        import deeplake
        
        # Load Animal10n dataset from Deep Lake
        try:
            ds_train = deeplake.open_read_only('hub://activeloop/animal10n-train')
            ds_test = deeplake.open_read_only('hub://activeloop/animal10n-test')
            print("Using Deep Lake 4.0 API")
        except AttributeError:
            ds_train = deeplake.load('hub://activeloop/animal10n-train')
            ds_test = deeplake.load('hub://activeloop/animal10n-test')
            print("Using Deep Lake 3.x API")
        
        trainset_size = len(ds_train)
        print(f"Animal10n train size: {trainset_size}")
        print(f"Animal10n test size: {len(ds_test)}")
        print("Animal10n classes: cat, lynx, wolf, coyote, cheetah, jaguar, chimpanzee, orangutan, hamster, guinea_pig")
        print("Note: Animal10n contains noisy labels with ~8% noise rate")
        
        sample_image = ds_train[0]['images']
        sample_label = ds_train[0]['labels']
        print(f"Original Animal10n image shape: {sample_image.shape}")
        print(f"Original Animal10n image type: {type(sample_image)}")
        print(f"Original Animal10n label type: {type(sample_label)}")
        print(f"Sample label value: {sample_label}")
        
        # Test the numpy conversion
        try:
            test_img_numpy = sample_image.numpy()
            print(f"Converted image shape: {test_img_numpy.shape}, type: {type(test_img_numpy)}")
        except Exception as e:
            print(f"Error converting image to numpy: {e}")
            
        try:
            test_label_numpy = sample_label.numpy() if hasattr(sample_label, 'numpy') else sample_label
            print(f"Converted label: {test_label_numpy}, type: {type(test_label_numpy)}")
        except Exception as e:
            print(f"Error converting label: {e}")
        
        def animal10n_transform_train(sample):
            # Optimized transform pipeline - fewer conversions
            
            # Handle both Deep Lake tensor and numpy array cases
            image_data = sample['images']
            if hasattr(image_data, 'numpy'):
                image_numpy = image_data.numpy()
            else:
                image_numpy = image_data
            
            label_data = sample['labels']
            if hasattr(label_data, 'numpy'):
                label_value = label_data.numpy()
            else:
                label_value = label_data
            
            # Convert to PIL and apply transforms in one go
            from PIL import Image
            if image_numpy.dtype != np.uint8:
                image_numpy = (image_numpy * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_numpy)
            
            # Apply transforms directly
            pil_image = pil_image.resize((32, 32))
            pil_image = transforms.functional.crop(pil_image, 
                                                 torch.randint(0, 5, (1,)).item(),
                                                 torch.randint(0, 5, (1,)).item(), 32, 32)
            if torch.rand(1) > 0.5:
                pil_image = transforms.functional.hflip(pil_image)
            
            # Convert to tensor and normalize
            tensor_image = transforms.functional.to_tensor(pil_image)
            tensor_image = transforms.functional.normalize(tensor_image, 
                                                         mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])
            
            return {
                'images': tensor_image,
                'labels': label_value
            }
        
        def animal10n_transform_test(sample):
            # Optimized transform pipeline - fewer conversions
            
            # Handle both Deep Lake tensor and numpy array cases
            image_data = sample['images']
            if hasattr(image_data, 'numpy'):
                image_numpy = image_data.numpy()
            else:
                image_numpy = image_data
            
            label_data = sample['labels']
            if hasattr(label_data, 'numpy'):
                label_value = label_data.numpy()
            else:
                label_value = label_data
            
            # Convert to PIL and apply transforms in one go
            from PIL import Image
            if image_numpy.dtype != np.uint8:
                image_numpy = (image_numpy * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_numpy)
            pil_image = pil_image.resize((32, 32))
            
            # Convert to tensor and normalize
            tensor_image = transforms.functional.to_tensor(pil_image)
            tensor_image = transforms.functional.normalize(tensor_image,
                                                         mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])
            
            return {
                'images': tensor_image,
                'labels': label_value
            }

        # Use wrapper for training with CACHING enabled for speed
        train_dataset = Animal10nWrapper(ds_train, animal10n_transform_train, use_cache=True, return_index=True)
        
        # Increase workers back to normal for better performance
        animal_kwargs = {'num_workers': 4, 'pin_memory': True}  # Increased from 2
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.bs, 
            shuffle=True, 
            **animal_kwargs
        )
        
        # Use wrapper for validation without indexes for standard format
        val_dataset = Animal10nWrapper(ds_test, animal10n_transform_test, use_cache=False, return_index=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.bs,
            shuffle=False,
            **animal_kwargs
        )
    else:
        raise Exception("Wrong Dataset")

    print(f"Loading {args.id} with num classes = {num_classes}")
    
    # Update K parameter based on number of classes
    args.K = num_classes
    
    # Create model based on model_type argument
    if args.model_type == 'densenet':
        model = ood_detect.OOD_Detection(args.M, args.K, args.layers, 'densenet')
    elif args.model_type == 'resnet18':
        model = ood_detect.OOD_Detection(args.M, args.K, args.layers, 'resnet18', r=args.r)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    print(f'Using {args.model_type} model')

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    model = model.cuda()
    
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
    criterion = get_loss_function('gce', num_classes=num_classes, trainset_size=trainset_size) 
    #criterion = nn.CrossEntropyLoss().cuda()
    loss_func_ce = F.nll_loss
    optimizer = torch.optim.SGD(model.fnet.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
   
    optimizer_trans = torch.optim.Adam(model.trans.parameters(), lr=args.lr)
    scheduler_f = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.epochs)
        adjust_learning_rate(optimizer_trans, epoch, args.epochs)

        # Train for one epoch
        train(train_loader, model, loss_func_ce, optimizer, epoch, num_classes, criterion, args, scheduler_f, optimizer_trans)
        
        # Evaluate on validation set
        prec1 = validate(val_loader, model, loss_func_ce, epoch, criterion, args)

        # Remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        save_checkpoint(args, {
            'epoch': epoch + 1,  
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=f"checkpoint_{epoch}.pth.tar")

    print('Best accuracy: ', best_prec1)

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
    lambda_sparse = 0

    for i, batch in enumerate(train_loader):
        # Handle different batch formats
        if len(batch) == 3:
            input, target, indexes = batch
        elif len(batch) == 2:
            input, target = batch
            indexes = torch.arange(len(target))  # Create dummy indexes
        else:
            # Handle dict format (fallback, shouldn't happen with our wrapper)
            if args.id == "Animal10n" and isinstance(batch, dict):
                input = batch['images']
                target = batch['labels']
                indexes = torch.arange(len(target))
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")
        
        target = target.cuda()
        input = input.cuda()
        indexes = indexes.cuda()
        
        # Handle different target formats for Animal10n
        if args.id == "Animal10n":
            if target.dim() > 1:
                if target.shape[1] == 1:
                    target = target.squeeze(1)
                elif target.shape[1] > 1:
                    target = torch.argmax(target, dim=1)
            target = target.long()
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        indexes_var = torch.autograd.Variable(indexes)

        final_output, reg_loss = model(input_var)
        ce_loss = criterion(final_output, target_var,indexes)  # Now includes indexes
        loss = ce_loss + lambda_sparse * reg_loss
        prec1 = accuracy(final_output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    batch_time = AverageMeter()
    losses = AverageMeter()
    kl_loss_meter = AverageMeter()
    top1 = AverageMeter()
   
    batch_size = args.bs
    M = args.M
    K = args.K
    model.eval()
    lambda_sparse = 0
    end = time.time()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Handle different batch formats
            if len(batch) == 3:
                input, target, indexes = batch
            elif len(batch) == 2:
                input, target = batch
                indexes = torch.arange(len(target))  # Create dummy indexes
            else:
                # Handle dict format (shouldn't happen with our wrapper now)
                if args.id == "Animal10n" and isinstance(batch, dict):
                    input = batch['images']
                    target = batch['labels']
                    indexes = torch.arange(len(target))
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)} elements")
            
            target = target.cuda()
            input = input.cuda()
            indexes = indexes.cuda()
            
            if args.id == "Animal10n":
                if target.dim() > 1:
                    if target.shape[1] == 1:
                        target = target.squeeze(1)
                    elif target.shape[1] > 1:
                        target = torch.argmax(target, dim=1)
                target = target.long()
            
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)
            indexes_var = torch.autograd.Variable(indexes, volatile=True)

            final_output, reg_loss = model(input_var)
            ce_loss = criterion(final_output, target_var)  # Now includes indexes
            loss = ce_loss + lambda_sparse * reg_loss
            prec1 = accuracy(final_output.data, target, topk=(1,))[0]

            top1.update(prec1, input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def save_checkpoint(args, state, is_best, filename):
    if args.model_type == 'densenet':
        if args.id == "Animal10n":
            directory = os.path.join("./checkpoints", args.id, "densenet_animal10n_noodle_sce")
        else:
            directory = os.path.join("./checkpoints", args.id, "densenet_cm_pi_rand2_v2")
    elif args.model_type == 'resnet18':
        if args.id == "Animal10n":
            directory = os.path.join("./checkpoints", args.id, "resnet18_animal10n")
        else:
            directory = os.path.join("./checkpoints", args.id, "resnet18_cm_pi_lamda_ema_loss_warmup")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.pth.tar'))

class AverageMeter(object):
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
    if tot_epochs == 300:
         lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    elif tot_epochs == 200:
         lr = args.lr * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 75)) * (0.1 ** (epoch // 90))
    elif tot_epochs == 100:
         lr = args.lr * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 75)) * (0.1 ** (epoch // 90))
    else:
        raise Exception("Check Epochs")
    
    print(f"Current lr: {lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
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