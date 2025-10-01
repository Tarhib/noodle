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
import utils.metrics_snn as metrics
import faiss
from tqdm import tqdm
from pathlib import Path
from types import MethodType
import models.ood_detect as ood_detect
from torch.utils.data import Dataset
import deeplake
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load CIFAR-10 noisy labels
noise_file = torch.load('./data/CIFAR-10_human.pt')
clean_label   = noise_file['clean_label']
worst_label   = noise_file['worse_label']
aggre_label   = noise_file['aggre_label']
random_label1 = noise_file['random_label1']
random_label2 = noise_file['random_label2']
random_label3 = noise_file['random_label3']

# Choose a noisy label version (modify as needed)
selected_noise_label = random_label3  # Change this to aggre_label, random_label1, etc.

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

# Animal-10N Dataset Functions using Deep Lake
def get_animal10n_loaders_ood(batch_size, transform_test):
    """Get Animal-10N data loaders using Deep Lake API for OOD detection"""
    print("Loading Animal-10N dataset using Deep Lake for OOD detection...")
    
    # Load Animal-10N dataset from Deep Lake with fallback
    try:
        ds_train = deeplake.open_read_only('hub://activeloop/animal10n-train')
        ds_test = deeplake.open_read_only('hub://activeloop/animal10n-test')
        print("Using Deep Lake 4.0 API")
    except AttributeError:
        ds_train = deeplake.load('hub://activeloop/animal10n-train')
        ds_test = deeplake.load('hub://activeloop/animal10n-test')
        print("Using Deep Lake 3.x API")
    
    print(f"Animal-10N train size: {len(ds_train)}")
    print(f"Animal-10N test size: {len(ds_test)}")
    print("Animal-10N classes: cat, lynx, wolf, coyote, cheetah, jaguar, chimpanzee, orangutan, hamster, guinea_pig")
    
    def animal10n_transform_ood(sample):
        """Transform function for OOD detection - no augmentation needed"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),  # Resize to 32x32 to match CIFAR
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Ensure labels are properly converted to tensors
        image = transform(sample['images'])
        label = sample['labels']
        
        # Convert label to tensor if it's not already
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()
        elif isinstance(label, (int, np.integer)):
            label = torch.tensor(label, dtype=torch.long)
        elif isinstance(label, str):
            try:
                label = torch.tensor(int(label), dtype=torch.long)
            except ValueError:
                label = torch.tensor(0, dtype=torch.long)
        elif not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        
        # Ensure label is in correct range (0-9)
        if label < 0 or label > 9:
            label = torch.clamp(label, 0, 9)
        
        return {
            'images': image,
            'labels': label  # Return as scalar, not with extra dimension
        }

    # Create data loaders
    train_loader = ds_train.pytorch(
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        transform=animal10n_transform_ood,
        decode_method={'images': 'numpy', 'labels': 'numpy'}
    )
    
    test_loader = ds_test.pytorch(
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        transform=animal10n_transform_ood,
        decode_method={'images': 'numpy', 'labels': 'numpy'}
    )
    
    return test_loader, train_loader

# Wrapper class to handle Animal-10N dict format for OOD detection
class Animal10nWrapper:
    def __init__(self, deeplake_loader):
        self.deeplake_loader = deeplake_loader
        self.dataset_size = len(deeplake_loader.dataset)
    
    def __iter__(self):
        for batch in self.deeplake_loader:
            if isinstance(batch, dict):
                # Convert dict format to tuple format for compatibility
                yield batch['images'], batch['labels']
            else:
                yield batch
    
    def __len__(self):
        return len(self.deeplake_loader)
    
    @property 
    def dataset(self):
        # Create a mock dataset object with the size attribute
        class MockDataset:
            def __init__(self, size):
                self.size = size
            def __len__(self):
                return self.size
        return MockDataset(self.dataset_size)

def id_loader(args):
    in_dataset = args.in_dataset
    bs = args.bs

    # Set up normalization for each dataset
    if in_dataset == "CIFAR-10":
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    elif in_dataset == "CIFAR-100":
        normalize = transforms.Normalize(mean=[0.507,0.487,0.441], std=[0.267, 0.256, 0.276])
    elif in_dataset in ["Animal-10N", "Animal10n"]:  # Handle both naming conventions
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        # Default normalization if dataset not recognized
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        print(f"Warning: Unknown dataset {in_dataset}, using default ImageNet normalization")
    
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
        
        trainset_NoiseDataset = CIFAR10Noisy(trainset, selected_noise_label)
        trainloaderIn = torch.utils.data.DataLoader(trainset_NoiseDataset, batch_size=bs, shuffle=True, num_workers=2)
        num_classes = 10

    elif in_dataset == "CIFAR-100":
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True, num_workers=2)
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
        num_classes = 100
    
    elif in_dataset in ["Animal-10N", "Animal10n"]:  # Handle both naming conventions
        # Load Animal-10N using Deep Lake
        testloader_raw, trainloader_raw = get_animal10n_loaders_ood(bs, transform_test)
        
        # Wrap the loaders to handle dict format
        testloaderIn = Animal10nWrapper(testloader_raw)
        trainloaderIn = Animal10nWrapper(trainloader_raw)
        num_classes = 10
        
        print(f"Animal-10N dataset loaded: train={len(trainloaderIn.dataset)}, test={len(testloaderIn.dataset)}")
    
    else:
        raise ValueError(f"Unsupported dataset: {in_dataset}")
    
    args.num_classes = num_classes
    args.transform_test = transform_test
    args.transform_fashion_mnist = transform_fashion_mnist
    return testloaderIn, trainloaderIn, args


def model_loader(args):
    model_arch = args.model_arch
    num_classes = args.num_classes
    
    if model_arch == 'densenet':
        print("Loading DenseNet OOD Detection model...")
        model = ood_detect.OOD_Detection(args.M, args.K, args.layers, 'densenet')
        model = model.cuda()
        
        # Normalize the dataset name for checkpoint path
        normalized_dataset = "Animal10n" if args.in_dataset in ["Animal-10N", "Animal10n"] else args.in_dataset
        checkpoint_path = "./checkpoints/{in_dataset}/densenet/model_best.pth.tar".format(in_dataset=normalized_dataset)
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path)
        except FileNotFoundError:
            print(f"Checkpoint file not found: {checkpoint_path}")
            raise
        
        # Handle different checkpoint formats
        state_dict = None
        
        # Check available keys in checkpoint
        available_keys = list(checkpoint.keys())
        print(f"Available keys in checkpoint: {available_keys}")
        
        if 'state_dict' in checkpoint:
            # Standard format (our updated training script includes this)
            print("Using 'state_dict' key")
            state_dict = checkpoint['state_dict']
        elif 'net1_state_dict' in checkpoint:
            # DivideMix format - use first network (primary network)
            print("Using 'net1_state_dict' key")
            state_dict = checkpoint['net1_state_dict']
        elif 'net2_state_dict' in checkpoint:
            # DivideMix format - use second network as fallback
            print("Using 'net2_state_dict' key")
            state_dict = checkpoint['net2_state_dict']
        else:
            raise KeyError(f"No compatible state dict found. Available keys: {available_keys}")
        
        print(f"Original state dict keys: {len(state_dict)}")
        
        # Load the state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
            
        # Print checkpoint info if available
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_acc' in checkpoint:
            print(f"Best accuracy: {checkpoint['best_acc']:.2f}%")
        elif 'accuracy' in checkpoint:
            print(f"Accuracy: {checkpoint['accuracy']:.2f}%")
       
    elif model_arch == 'resnet50':
        print("Loading ResNet50 model...")
        model = resnet.ResNet50(num_class=num_classes)
        # Normalize the dataset name for checkpoint path
        normalized_dataset = "Animal10n" if args.in_dataset in ["Animal-10N", "Animal10n"] else args.in_dataset
        checkpoint_path = "./checkpoints/{in_dataset}/resnet50/model_best.pth.tar".format(in_dataset=normalized_dataset)
        
        try:
            checkpoint = torch.load(checkpoint_path)
        except FileNotFoundError:
            print(f"Checkpoint file not found: {checkpoint_path}")
            raise
            
        if 'net' in checkpoint:
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['net'].items()}
        elif 'state_dict' in checkpoint:
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        else:
            raise KeyError(f"No compatible state dict found for ResNet50. Available keys: {list(checkpoint.keys())}")
            
        model.load_state_dict(state_dict)
    
    elif model_arch == 'resnet18':
        print("Loading ResNet18 OOD Detection model...")
        model = ood_detect.OOD_Detection(args.M, args.K, args.layers, 'resnet18', r=0.05)
        
        # Normalize the dataset name for checkpoint path
        normalized_dataset = "Animal10n" if args.in_dataset in ["Animal-10N", "Animal10n"] else args.in_dataset
        checkpoint_path = "./checkpoints/{in_dataset}/resnet18/model_best.pth.tar".format(in_dataset=normalized_dataset)
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path)
        except FileNotFoundError:
            print(f"Checkpoint file not found: {checkpoint_path}")
            raise
            
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
    
    else:
        assert False, 'Not supported model arch: {}'.format(model_arch)
    
    model.cuda()
    model.eval()
    print("Model loaded successfully!")
    return model

def get_out_loader(out_dataset, args):
    batch_size = args.bs
    transform = args.transform_test
    transform_fashion_mnist = args.transform_fashion_mnist
    
    if out_dataset == 'SVHN':
        testsetout = torchvision.datasets.SVHN('./ood_data/', split='test', transform=transform, download=True)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == 'FashionMNIST':
        testsetout = torchvision.datasets.FashionMNIST(
            './ood_data/',
            train=False,  # Set to False to load the test set
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
    elif out_dataset in ['Animal-10N', 'Animal10n']:  # Handle both naming conventions
        # Use Animal-10N test set as OOD data for other datasets
        print(f"Loading Animal-10N as OOD dataset...")
        testloader_raw, _ = get_animal10n_loaders_ood(batch_size, transform)
        testloaderOut = Animal10nWrapper(testloader_raw)
    else:
        testsetout = torchvision.datasets.ImageFolder("./ood_data/{}".format(out_dataset), transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    return testloaderOut

def run_mahalanobis_func(in_dataset, model_arch, out_datasets, start, end):
    """
    Replace KNN with Mahalanobis distance for OOD detection on encoded features
    """
    print("Running Mahalanobis distance-based OOD detection on encoded features...")
    
    # Load training features
    cache_name = f"cache/{in_dataset}_{model_arch}_train_in_alllayers.npy"
    feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
    feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
    label_log = label_log.astype(np.int32)
    
    class_num = score_log.shape[1]
    print(f"Number of classes: {class_num}")
    
    # Load validation features
    cache_name = f"cache/{in_dataset}_{model_arch}_val_in_alllayers.npy"
    feat_log_val, score_log_val, label_log_val = np.load(cache_name, allow_pickle=True)
    feat_log_val, score_log_val = feat_log_val.T.astype(np.float32), score_log_val.T.astype(np.float32)
    
    # Load OOD features
    ood_feat_log_all = {}
    for ood_dataset in out_datasets:
        cache_name = f"cache/{ood_dataset}vs{in_dataset}_{model_arch}_out_alllayers.npy"
        ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
        ood_feat_log, ood_score_log = ood_feat_log.T.astype(np.float32), ood_score_log.T.astype(np.float32)
        ood_feat_log_all[ood_dataset] = ood_feat_log
    
    # Feature preprocessing (same as KNN version)
    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(start, end)]))
    
    # Extract features from the specified range
    ftrain = prepos_feat(feat_log)  # Training features
    ftest = prepos_feat(feat_log_val)  # ID test features
    
    print(f"Feature dimensions: {ftrain.shape[1]}")
    print(f"Training samples: {ftrain.shape[0]}")
    print(f"Test samples: {ftest.shape[0]}")
    
    # Prepare OOD features
    food_all = {}
    for ood_dataset in out_datasets:
        food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])
        print(f"OOD dataset {ood_dataset}: {food_all[ood_dataset].shape[0]} samples")
    
    # ===== MAHALANOBIS DISTANCE CALCULATION =====
    
    # 1. Calculate class-wise means
    print("Calculating class-wise means...")
    classwise_mean = np.zeros((class_num, ftrain.shape[1]))
    
    for class_id in range(class_num):
        class_mask = (label_log == class_id)
        if np.sum(class_mask) > 0:
            classwise_mean[class_id] = np.mean(ftrain[class_mask], axis=0)
            print(f"Class {class_id}: {np.sum(class_mask)} samples")
        else:
            print(f"Warning: No samples found for class {class_id}")
    
    # 2. Calculate covariance matrix
    print("Calculating covariance matrix...")
    # Center the features
    overall_mean = np.mean(ftrain, axis=0)
    centered_features = ftrain - overall_mean
    
    # Calculate covariance matrix
    cov_matrix = np.cov(centered_features.T)
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    
    # 3. Calculate precision matrix (inverse of covariance)
    print("Calculating precision matrix...")
    try:
        cov_matrix_reg = cov_matrix
        precision_matrix = np.linalg.inv(cov_matrix_reg)
        cond_num = np.linalg.cond(precision_matrix)
        print(f"Condition number: {cond_num}")
    except np.linalg.LinAlgError:
        print("Using pseudo-inverse due to singular matrix")
        precision_matrix = np.linalg.pinv(cov_matrix)
        cond_num = np.linalg.cond(precision_matrix)
        print(f"Condition number (pseudo-inverse): {cond_num}")
    
    # 4. Function to calculate Mahalanobis scores
    def get_mahalanobis_scores(features, classwise_mean, precision_matrix):
        """
        Calculate Mahalanobis scores for given features
        Returns the maximum score across all classes for each sample
        """
        scores = []
        
        for i in tqdm(range(features.shape[0]), desc="Computing Mahalanobis scores"):
            sample_scores = []
            for class_id in range(class_num):
                # Calculate difference from class mean
                diff = features[i] - classwise_mean[class_id]
                # Calculate Mahalanobis distance
                mahal_dist = -0.5 * np.dot(np.dot(diff, precision_matrix), diff.T)
                sample_scores.append(mahal_dist)
            
            # Take maximum score across all classes
            max_score = np.max(sample_scores)
            scores.append(max_score)
        
        return np.array(scores)
    
    # 5. Calculate scores for ID test data
    print("Calculating Mahalanobis scores for ID test data...")
    scores_in = get_mahalanobis_scores(ftest, classwise_mean, precision_matrix)
    
    print(f"ID scores - Mean: {np.mean(scores_in):.4f}, Std: {np.std(scores_in):.4f}")
    
    # 6. Calculate scores for each OOD dataset and evaluate
    all_results = []
    all_score_ood = []
    
    for ood_dataset, food in food_all.items():
        print(f"\nEvaluating OOD dataset: {ood_dataset}")
        scores_ood_test = get_mahalanobis_scores(food, classwise_mean, precision_matrix)
        
        print(f"OOD scores - Mean: {np.mean(scores_ood_test):.4f}, Std: {np.std(scores_ood_test):.4f}")
        
        all_score_ood.extend(scores_ood_test)
        
        # Calculate metrics
        results = metrics.cal_metric(scores_in, scores_ood_test)
        all_results.append(results)
        
        # Print individual results - handle different metrics formats
        print("Available metrics keys:", list(results.keys()) if isinstance(results, dict) else "Not a dict")
        
        if isinstance(results, dict):
            # Handle different possible key formats
            auroc_key = None
            aupr_key = None
            fpr95_key = None
            
            # Find the right keys (case insensitive)
            for key in results.keys():
                key_lower = str(key).lower()
                if 'auroc' in key_lower or 'auc' in key_lower:
                    auroc_key = key
                elif 'aupr' in key_lower or 'aupr' in key_lower:
                    aupr_key = key
                elif 'fpr95' in key_lower or 'fpr' in key_lower:
                    fpr95_key = key
            
            # Print results with available keys
            auroc_val = results.get(auroc_key, 'N/A') if auroc_key else 'N/A'
            aupr_val = results.get(aupr_key, 'N/A') if aupr_key else 'N/A'
            fpr95_val = results.get(fpr95_key, 'N/A') if fpr95_key else 'N/A'
            
            print(f"AUROC: {auroc_val}, AUPR: {aupr_val}, FPR95: {fpr95_val}")
            
            # Also print all available metrics
            print("All metrics:", {k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in results.items()})
        else:
            print(f"Results: {results}")
    
    # Print overall results
    print("\n" + "="*50)
    metrics.print_all_results(all_results, out_datasets, 'Mahalanobis Distance (Encoded)')
    print("="*50)
    
    return all_results

def run_knn_func(in_dataset, model_arch, out_datasets, start, end):
    """
    KNN function for encoded features
    """
    print("Running KNN-based OOD detection on encoded features...")
    
    in_dataset = in_dataset
    model_arch = model_arch
    cache_name = f"cache/{in_dataset}_{model_arch}_train_in_alllayers.npy"
    feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
    feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
    class_num = score_log.shape[1]
    start = start; stop = end;
    cache_name = f"cache/{in_dataset}_{model_arch}_val_in_alllayers.npy"
    feat_log_val, score_log_val, label_log_val = np.load(cache_name, allow_pickle=True)
    feat_log_val, score_log_val = feat_log_val.T.astype(np.float32), score_log_val.T.astype(np.float32)
    ood_feat_log_all = {}
    for ood_dataset in out_datasets:
        cache_name = f"cache/{ood_dataset}vs{in_dataset}_{model_arch}_out_alllayers.npy"
        ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
        ood_feat_log, ood_score_log = ood_feat_log.T.astype(np.float32), ood_score_log.T.astype(np.float32)
        ood_feat_log_all[ood_dataset] = ood_feat_log
    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(start, end)]))
    ftrain = prepos_feat(feat_log)
    ftest = prepos_feat(feat_log_val)
    food_all = {}
    for ood_dataset in out_datasets:
        food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])
    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)
    for K in [20]:
        D, _ = index.search(ftest, K)
        scores_in = -D[:,-1]
        all_results = []
        all_score_ood = []
        for ood_dataset, food in food_all.items():
            D, _ = index.search(food, K)
            scores_ood_test = -D[:,-1]
            all_score_ood.extend(scores_ood_test)
            results = metrics.cal_metric(scores_in, scores_ood_test)
            all_results.append(results)
        metrics.print_all_results(all_results, out_datasets, f'SNN k={K} (Encoded)')
        print()

# Modified feat_extract function to work with encoded features
def feat_extract(args, use_mahalanobis=False):
    """
    Feature extraction function for encoded features (h_encoded from OOD Detection model)
    
    Args:
        args: Arguments object containing dataset and model configuration
        use_mahalanobis: If True, use Mahalanobis distance; if False, use KNN (default: False)
    """
    
    FORCE_RUN = True
    testloaderIn, trainloaderIn, args = id_loader(args)
    
    print(f"{args.in_dataset} with {args.num_classes} classes")
    model = model_loader(args)

    dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
    score, h_encoded_dummy = model.feature_list(dummy_input)  # Updated: output is now h_encoded
    B, M = h_encoded_dummy.shape

    # Calculate feature dimensions
    featdims = [M]  # Flatten the (M, K) dimensions for each batch
    start = 0
    end = featdims[0]
    print(f"Feature dimensions: {featdims}")
    print(f"Start: {start}, End: {end}")
    
    begin = time.time()
    num_classes = args.num_classes
    batch_size = args.bs
    
    # Create cache directory if it doesn't exist
    os.makedirs("cache", exist_ok=True)
    
    # Feature extraction for ID training data
    for split, in_loader in [('train', trainloaderIn)]:
        cache_name = f"cache/{args.in_dataset}_{args.model_arch}_{split}_in_alllayers.npy"
        if FORCE_RUN or not os.path.exists(cache_name):
            feat_log = np.zeros((len(in_loader.dataset), sum(featdims)))
            score_log = np.zeros((len(in_loader.dataset), num_classes))
            label_log = np.zeros(len(in_loader.dataset))
            
            # Track unique classes per batch
            classes_per_batch = []
            
            model.eval()
            for batch_idx, batch_data in enumerate(in_loader):
                # Handle different data formats
                if len(batch_data) == 2:
                    inputs, targets = batch_data
                elif len(batch_data) > 2:
                    inputs, targets = batch_data[0], batch_data[1]
                else:
                    print(f"Warning: Unexpected batch format with {len(batch_data)} elements")
                    continue
                    
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))
                
                # Handle target shape for Animal-10N (squeeze extra dimensions) - BEFORE using targets
                if args.in_dataset in ["Animal-10N", "Animal10n"] and targets.dim() > 1:
                    print(f"Original targets shape: {targets.shape}")
                    if targets.shape[1] == 1:
                        targets = targets.squeeze(1)
                    elif targets.shape[1] > 1:
                        targets = torch.argmax(targets, dim=1)
                    print(f"After processing targets shape: {targets.shape}")
                
                # Ensure targets is 1D
                if targets.dim() > 1:
                    if targets.shape[1] == 1:
                        targets = targets.squeeze(1)
                    else:
                        targets = torch.argmax(targets, dim=1)
                        
                # Double check targets shape
                if batch_idx == 0:  # Print once for debugging
                    print(f"Final targets shape: {targets.shape}")
                    print(f"Expected label_log shape for this batch: {(end_ind - start_ind,)}")
                
                # Count unique classes in this batch
                unique_classes_in_batch = len(torch.unique(targets))
                classes_per_batch.append(unique_classes_in_batch)
                
                score, h_encoded = model.feature_list(inputs)  # h_encoded shape: (B, M, K)
                # Flatten h_encoded from (B, M, K) to (B, M * K)
                out = h_encoded.view(h_encoded.size(0), -1)
                
                feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                label_log[start_ind:end_ind] = targets.data.cpu().numpy()
                score_log[start_ind:end_ind] = score.data.cpu().numpy()
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(in_loader)} processed")
            
            # Calculate and print average classes per batch
            avg_classes_per_batch = np.mean(classes_per_batch)
            print(f"Train - Average classes per batch: {avg_classes_per_batch:.2f}")
            print(f"Train - Min classes per batch: {min(classes_per_batch)}")
            print(f"Train - Max classes per batch: {max(classes_per_batch)}")
            
            np.save(cache_name, np.array([feat_log.T, score_log.T, label_log], dtype=object))
        else:
            print(f"Loading cached features from {cache_name}")
            feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
            feat_log, score_log = feat_log.T, score_log.T
    
    # Feature extraction for ID validation data
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
                
                # Handle target shape for Animal-10N (squeeze extra dimensions)
                if args.in_dataset in ["Animal-10N", "Animal10n"] and targets.dim() > 1:
                    if targets.shape[1] == 1:
                        targets = targets.squeeze(1)
                    elif targets.shape[1] > 1:
                        targets = torch.argmax(targets, dim=1)
                
                # Ensure targets is 1D
                if targets.dim() > 1:
                    if targets.shape[1] == 1:
                        targets = targets.squeeze(1)
                    else:
                        targets = torch.argmax(targets, dim=1)
                
                # Count unique classes in this batch
                unique_classes_in_batch = len(torch.unique(targets))
                classes_per_batch.append(unique_classes_in_batch)
                
                score, h_encoded = model.feature_list_val(inputs)  # h_encoded shape: (B, M, K)
                # Flatten h_encoded from (B, M, K) to (B, M * K)
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
            print(f"Loading cached features from {cache_name}")
            feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
            feat_log, score_log = feat_log.T, score_log.T
    
    # Feature extraction for OOD data
    d = ['SVHN', 'FashionMNIST','LSUN', 'iSUN', 'dtd', 'places365']
    
    # Add Animal-10N to OOD datasets if current dataset is not Animal-10N
    if args.in_dataset not in ["Animal-10N", "Animal10n"]:
        d.append('Animal-10N')
    
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

                # Flatten h_encoded from (B, M, K) to (B, M * K)
                out = h_encoded.view(h_encoded.size(0), -1)

                ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                ood_score_log[start_ind:end_ind] = score.data.cpu().numpy()

                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(out_loader)} processed")
            
            np.save(cache_name, np.array([ood_feat_log.T, ood_score_log.T], dtype=object))
        else:
            print(f"Loading cached OOD features from {cache_name}")
            ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
            ood_feat_log, ood_score_log = ood_feat_log.T, ood_score_log.T
    
    print(f"Feature extraction completed in {time.time() - begin:.2f} seconds")
    
    # Choose evaluation method based on flag
    if use_mahalanobis:
        print("\n" + "="*50)
        print("Running Mahalanobis Distance-based OOD Detection on Encoded Features")
        print("="*50)
        run_mahalanobis_func(args.in_dataset, args.model_arch, d, start, end)
    else:
        print("\n" + "="*50)
        print("Running KNN-based OOD Detection on Encoded Features")
        print("="*50)
        run_knn_func(args.in_dataset, args.model_arch, d, start, end)


# Example usage:
# 
# # For KNN evaluation (default):
# feat_extract(args, use_mahalanobis=False)
# 
# # For Mahalanobis evaluation:
# feat_extract(args, use_mahalanobis=True)
#
# # Example with argument setup:
# class Args:
#     def __init__(self):
#         self.in_dataset = "Animal10n"   # Can be "CIFAR-10", "CIFAR-100", "Animal-10N", or "Animal10n"
#         self.model_arch = "densenet"    # Can be "densenet", "resnet18", or "resnet50"
#         self.bs = 64
#         self.device = torch.device("cuda")
#         self.M = 6  # Number of annotators
#         self.K = 10  # Number of classes
#         self.layers = 100  # DenseNet layers
#         self.noise_type = "symmetric"
#         self.noise_rate = 0.1
#
# args = Args()
#
# # Run with KNN on Animal-10N
# feat_extract(args, use_mahalanobis=False)
#
# # Run with Mahalanobis on Animal-10N
# feat_extract(args, use_mahalanobis=True)

"""
Usage Examples:

1. Run OOD detection on Animal-10N with DenseNet:
   python ood_script.py --in_dataset Animal10n --model_arch densenet
   OR
   python ood_script.py --in_dataset Animal-10N --model_arch densenet

2. Run OOD detection on CIFAR-10 with ResNet18:
   python ood_script.py --in_dataset CIFAR-10 --model_arch resnet18

3. Run OOD detection on CIFAR-100 with DenseNet:
   python ood_script.py --in_dataset CIFAR-100 --model_arch densenet

Key Features:
- Remote Animal-10N dataset loading via Deep Lake (no local files needed)
- Supports both "Animal-10N" and "Animal10n" naming conventions
- Support for both KNN and Mahalanobis distance-based OOD detection
- Handles different data formats (dict from Deep Lake, tuple from standard datasets)
- Animal-10N images are resized to 32x32 to match CIFAR for consistency
- Automatic handling of target tensor shapes for Animal-10N
- The script expects trained model checkpoints in: ./checkpoints/Animal10n/densenet/model_best.pth.tar

Requirements:
- pip install deeplake
- pip install faiss-gpu (or faiss-cpu)
- Trained model checkpoints in the expected directory structure
"""