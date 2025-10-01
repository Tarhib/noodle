from pickle import FALSE
import torch
import os
import numpy as np
import torch.nn.functional as F
import time
import torchvision
import torchvision.transforms as transforms
import models.densenet as dn
import models.resnet as resnet
import numpy as np
import time
from run_snn import run_knn_func
from pathlib import Path
from types import MethodType
import models.ood_detect as ood_detect
from data.cifar import CIFAR10, CIFAR100
from torch.utils.data import Dataset 

# Load CIFAR-10 noisy labels
noise_file = torch.load('./data/CIFAR-10_human.pt')
clean_label   = noise_file['clean_label']
worst_label   = noise_file['worse_label']
aggre_label   = noise_file['aggre_label']
random_label1 = noise_file['random_label1']
random_label2 = noise_file['random_label2']
random_label3 = noise_file['random_label3']

# Choose a noisy label version (modify as needed)
selected_noise_label = aggre_label # Change this to aggre_label, random_label1, etc.

# Custom Dataset to Apply Noisy Labels
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

def id_loader(args):
    in_dataset = args.in_dataset
    bs = args.bs

    if in_dataset == "CIFAR-10":
            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    elif in_dataset == "CIFAR-100":
            normalize = transforms.Normalize(mean = [0.507,0.487,0.441], std = [0.267, 0.256, 0.276])
    
    transform_test = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                normalize,
            ])
    transform_fashion_mnist = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize,
            ])

    if in_dataset == "CIFAR-10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True, num_workers=2)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
        trainset_NoiseDataset= CIFAR10Noisy(trainset, selected_noise_label)
        trainloaderIn = torch.utils.data.DataLoader(trainset_NoiseDataset, batch_size=bs, shuffle=True, num_workers=2)
        num_classes = 10

    elif in_dataset == "CIFAR-100":
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True, num_workers=2)
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
        num_classes = 100
    
    args.num_classes = num_classes
    args.transform_test = transform_test
    args.transform_fashion_mnist = transform_fashion_mnist
    return testloaderIn, trainloaderIn, args

def model_loader(args):
    model_arch = args.model_arch
    num_classes = args.num_classes
    if model_arch == 'densenet':
        print("Densenet")
        model = ood_detect.OOD_Detection( args.M, args.K, args.layers, 'densenet')
        
        checkpoint = torch.load(
            "./checkpoints/{in_dataset}/densenet/model_best.pth.tar".format(in_dataset=args.in_dataset))
        state_dict = checkpoint['state_dict']
       
        model.load_state_dict(state_dict, strict = False)

    elif model_arch == 'resnet50':
        model = resnet.ResNet50(num_class=num_classes)
        checkpoint = torch.load(
                "./checkpoints/{in_dataset}/resnet50/model_best.pth.tar".format(in_dataset=args.in_dataset))
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['net'].items()}
        model.load_state_dict(state_dict)

    elif model_arch == 'resnet18':
        print("ResNet18")
        model = ood_detect.OOD_Detection(args.M, args.K, args.layers, 'resnet18')
        
        checkpoint = torch.load(
            "./checkpoints/{in_dataset}/resnet18/model_best.pth.tar".format(in_dataset=args.in_dataset))
        state_dict = checkpoint['state_dict']
        
        model.load_state_dict(state_dict, strict=False)

    else:
        assert False, 'Not supported model arch: {}'.format(model_arch)
    
    model.cuda()
    model.eval()
    return model

def get_out_loader(out_dataset, args):
    batch_size = args.bs
    transform = args.transform_test
    transform_fashion_mnist= args.transform_fashion_mnist
    if out_dataset == 'SVHN':
        testsetout = torchvision.datasets.SVHN('./ood_data/', split='test', transform=transform, download=True)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == 'FashionMNIST':
        testsetout = torchvision.datasets.FashionMNIST(
            './ood_data/',
            train=False,
            transform=transform_fashion_mnist,
            download=True
        )
        testloaderOut = torch.utils.data.DataLoader(
            testsetout,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
          )
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root="./ood_data/dtd/images", transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == 'places365':
        testsetout = torchvision.datasets.ImageFolder(root="./ood_data/Places365/", transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == 'CIFAR-10':
        testsetout = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == 'CIFAR-100':
        testsetout = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        testsetout = torchvision.datasets.ImageFolder("./ood_data/{}".format(out_dataset), transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    return testloaderOut

def feat_extract(args):
    FORCE_RUN = True
    testloaderIn, trainloaderIn, args = id_loader(args)
    
    print(f"{args.in_dataset} with {args.num_classes} classes")
    model = model_loader(args)

    dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
    score, h_encoded_dummy,_ = model.feature_list(dummy_input)
    B, M = h_encoded_dummy.shape

    # Calculate feature dimensions
    featdims = [M]
    start = 0
    end = featdims[0]
    print(f"Feature dimensions: {featdims}")
    print(f"Start: {start}, End: {end}")

    begin = time.time()
    num_classes = args.num_classes
    batch_size = args.bs

    for split, in_loader in [('train', trainloaderIn)]:
        cache_name = f"cache/{args.in_dataset}_{args.model_arch}_{split}_in_alllayers.npy"
        if FORCE_RUN or not os.path.exists(cache_name):
            # Pre-allocate based on expected sample count (28 per batch)
            samples_per_batch = 28  # Your model returns 28 pre-selected samples
            estimated_total_samples = len(in_loader) * samples_per_batch
            
            feat_log = np.zeros((estimated_total_samples, sum(featdims)))
            score_log = np.zeros((estimated_total_samples, num_classes))
            label_log = np.zeros(estimated_total_samples)
            
            # Track unique classes per batch
            classes_per_batch = []
            
            model.eval()
            ptr = 0  # Pointer to track current position in arrays
            
            for batch_idx, (inputs, targets, *others) in enumerate(in_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                
                # Count unique classes in this batch
                unique_classes_in_batch = len(torch.unique(targets))
                classes_per_batch.append(unique_classes_in_batch)
                
                score, h_encoded2, h_encoded = model.feature_list(inputs)
                
                # h_encoded contains the pre-selected samples (typically 28)
                actual_samples = h_encoded.size(0)
                
                # Flatten h_encoded
                out = h_encoded.view(h_encoded.size(0), -1)
                
                # Store using pointer-based indexing
                end_ptr = ptr + actual_samples
                
                if end_ptr <= len(feat_log):  # Safety check
                    feat_log[ptr:end_ptr, :] = out.data.cpu().numpy()
                    label_log[ptr:end_ptr] = targets.data[:actual_samples].cpu().numpy()
                    score_log[ptr:end_ptr] = score.data[:actual_samples].cpu().numpy()
                    ptr = end_ptr
                else:
                    print(f"Warning: Array capacity reached at batch {batch_idx}")
                    break
                
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(in_loader)} processed - stored {actual_samples} samples")
            
            # Trim arrays to actual size used
            feat_log = feat_log[:ptr]
            label_log = label_log[:ptr]
            score_log = score_log[:ptr]
            
            # Calculate and print statistics
            avg_classes_per_batch = np.mean(classes_per_batch)
            print(f"Train - Average classes per batch: {avg_classes_per_batch:.2f}")
            print(f"Train - Min classes per batch: {min(classes_per_batch)}")
            print(f"Train - Max classes per batch: {max(classes_per_batch)}")
            print(f"Train - Total samples stored: {ptr}")
            
            np.save(cache_name, np.array([feat_log.T, score_log.T, label_log], dtype=object))
        else:
            feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
            feat_log, score_log = feat_log.T, score_log.T

    for split, in_loader in [('val', testloaderIn)]:
        cache_name = f"cache/{args.in_dataset}_{args.model_arch}_{split}_in_alllayers.npy"
        if FORCE_RUN or not os.path.exists(cache_name):
            feat_log = np.zeros((len(in_loader.dataset), sum(featdims)))
            score_log = np.zeros((len(in_loader.dataset), num_classes))
            label_log = np.zeros(len(in_loader.dataset))
            
            # Track unique classes per batch
            classes_per_batch = []
            
            model.eval()
            for batch_idx, (inputs, targets) in enumerate(in_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))
                
                # Count unique classes in this batch
                unique_classes_in_batch = len(torch.unique(targets))
                classes_per_batch.append(unique_classes_in_batch)
                
                score, h_encoded = model.feature_list_val(inputs)
                # Flatten h_encoded
                out = h_encoded.view(h_encoded.size(0), -1)
                feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                label_log[start_ind:end_ind] = targets.data.cpu().numpy()
                score_log[start_ind:end_ind] = score.data.cpu().numpy()
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(in_loader)} processed")
            
            # Calculate and print average classes per batch
            avg_classes_per_batch = np.mean(classes_per_batch)
            print(f"Validation - Average classes per batch: {avg_classes_per_batch:.2f}")
            print(f"Validation - Min classes per batch: {min(classes_per_batch)}")
            print(f"Validation - Max classes per batch: {max(classes_per_batch)}")
            
            np.save(cache_name, np.array([feat_log.T, score_log.T, label_log], dtype=object))
        else:
            feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
            feat_log, score_log = feat_log.T, score_log.T
            
    # Process OOD datasets
    d = ['SVHN', 'FashionMNIST','LSUN', 'iSUN', 'dtd', 'places365']
    for ood_dataset in d:
        out_loader = get_out_loader(ood_dataset, args)
        cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.model_arch}_out_alllayers.npy"
        if FORCE_RUN or not os.path.exists(cache_name):
            ood_feat_log = np.zeros((len(out_loader.dataset), sum(featdims)))
            ood_score_log = np.zeros((len(out_loader.dataset), num_classes))

            model.eval()
            for batch_idx, (inputs, _) in enumerate(out_loader):
                inputs = inputs.to(args.device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

                # Forward pass
                score, h_encoded = model.feature_list_val(inputs)

                # Flatten h_encoded
                out = h_encoded.view(h_encoded.size(0), -1)

                ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                ood_score_log[start_ind:end_ind] = score.data.cpu().numpy()

                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(out_loader)} processed")

            np.save(cache_name, np.array([ood_feat_log.T, ood_score_log.T], dtype=object))
        else:
            ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
            ood_feat_log, ood_score_log = ood_feat_log.T, ood_score_log.T

    print(f"Feature extraction completed in {time.time() - begin:.2f} seconds")
    run_knn_func(args.in_dataset, args.model_arch, d, start, end)