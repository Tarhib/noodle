
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
from run_subspace import run_ood_detection_with_confusion_projection
from pathlib import Path
from types import MethodType
import models.ood_detect as ood_detect
import torch.nn as nn

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
                # Optionally add normalization if needed
                # transforms.Normalize(mean=[...], std=[...]),
            ])

    if in_dataset == "CIFAR-10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True, num_workers=2)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
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
        #model = dn.DenseNet3(100, num_classes, growth_rate= 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=None)
        model = ood_detect.OOD_Detection( args.M, args.K, args.layers, 'densenet')
        model = model.cuda()
        checkpoint = torch.load(
            "./checkpoints/{in_dataset}/densenet/model_best.pth.tar".format(in_dataset=args.in_dataset))
        state_dict = checkpoint['state_dict']
        #state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fnet.fc.') and not k.startswith('linear.')}
        model.load_state_dict(state_dict, strict = False)
       
        #state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.') and not k.startswith('trans.')}
    
       


    elif model_arch == 'resnet50':
        model = resnet.ResNet50(num_class=num_classes)
        checkpoint = torch.load(
                "./checkpoints/{in_dataset}/resnet50/model_best.pth.tar".format(in_dataset=args.in_dataset))
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['net'].items()}
        model.load_state_dict(state_dict)

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
    else:
        testsetout = torchvision.datasets.ImageFolder("./ood_data/{}".format(out_dataset), transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    return testloaderOut

def feat_extract_subspace(args):
    FORCE_RUN = True
    testloaderIn, trainloaderIn, args = id_loader(args)

    print(f"{args.in_dataset} with {args.num_classes} classes")
    model = model_loader(args)

    dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
    model.cuda()
    
    final_output,B_matrix , h_sftmax,reg_loss = model(dummy_input) #B_matrix has shape K,MK
    
    

    b,M,K= args.bs,args.M,args.K
    featdims = [M*K]
    
    print(B_matrix)
   
    start = sum(featdims[:-1])
    end = start + featdims[-1]
    print(featdims)
    print(start, end)

    begin = time.time()
    num_classes = args.num_classes
    batch_size = args.bs

    '''
    def project_features(h_softmax, trans_matrices):
        # h_softmax: [batch, M, K]
        # trans_matrices: [M, K, K]
        device = h_softmax.device  # Get the device of h_softmax
        trans_matrices = trans_matrices.to(device)  # Move trans_matrices to the same device
        
        K = trans_matrices.shape[1]  # Get K from trans_matrices shape
        d = 3  # Reduced dimension (d < K)
        
        # Generate random projection matrix [K, d] and move to the same device
        random_projection = torch.randn(K, d).to(device)  # Random matrix to reduce dimension
        
        # Reduce dimension: [M, K, K] -> [M, K, d]
        trans_matrices = torch.einsum('mki,kd->mid', trans_matrices, random_projection)  # [M, K, d]
        
        print("Reduced trans_matrices shape:", trans_matrices.shape)  # Expected: [M, K, d]


       # Compute A^T
        A_transpose = trans_matrices.transpose(-1, -2)  # (M, K, K)
    
        # Compute A^T A
        ATA = torch.einsum('mij,mjk->mik', A_transpose, trans_matrices)  # (M, K, K)
    
        # Since A is invertible, (A^T A) is invertible
        ATA_inv = torch.linalg.inv(ATA)  # (M, K, K)
    
        # Compute P = A (A^T A)^{-1} A^T
        # (mij,mjk,mkl->mil)
        projection_matrices = torch.einsum('mij,mjk,mkl->mil', 
                                           trans_matrices, 
                                           ATA_inv, 
                                           A_transpose)  # (M, K, K)
        diff = torch.norm(torch.matmul(projection_matrices, projection_matrices) - projection_matrices)
        print("||P^2 - P||:", diff.item())
        
        # Project features
        projected_features = torch.einsum('bmj,mjk->bmk', h_softmax, projection_matrices)




    
    
        # Compute L2 distance between original and projected features
        # Flatten the last two dims or norm over them directly
        distances = torch.norm(h_softmax - projected_features, p=2, dim=(1, 2))  # [batch]

       
        
        #return distances
    
        return projected_features, distances
        
    '''
    def project_features(h_softmax, W,score,lambda_reg=1e-3):
      """
      Projects features using projection matrices and computes L2 distances.
  
      Args:
          h_softmax (torch.Tensor): [batch, M, K] feature tensor.
          W (torch.Tensor): [MK, K] transformation matrix.
          epsilon (float): Regularization for numerical stability.
  
      Returns:
          projected_features (torch.Tensor): [batch, M, K] projected features.
          distances (torch.Tensor): [batch] L2 distances between original and projected features.
      """
      device = h_softmax.device
      W = W.to(device)
      # Compute W^T
      W_transpose = W.transpose(-1, -2)  # [K, MK]
  
      # Compute W^T W with regularization
      WTW = torch.matmul(W_transpose, W)  # [K, K]
     
  
      # Invert W^T W
      WTW_inv = torch.linalg.inv(WTW)  # [K, K]
  
      # Compute P = W (W^T W)^{-1} W^T
      intermediate = torch.matmul(W, WTW_inv)  # [MK, K]
      projection_matrix = torch.matmul(intermediate, W_transpose)  # [MK, MK]
      
  
     
      h_softmax_flat = h_softmax
      #print("h_softmax_flat shape:", h_softmax_flat.shape)  # [M*K, K] = [50, 10]
      #print("projection_matrix shape:", projection_matrix.shape)  # [M*K, K] = [50, 10]
     

     
      # Perform the projection: [batch, MK] x [MK, MK] -> [batch, MK]
      projected_flat = torch.matmul(h_softmax_flat, projection_matrix)  # => [10, 50]

  
      # Reshape back to [batch, M, K]
      
      
      final_output= score
      final_output_T = final_output.transpose(0, 1)  # Shape: [K, batch]

      # Perform the matrix multiplication
      Wf_x = torch.matmul(W, final_output_T)  # Shape: [MK, batch]
      
      # Transpose back to get [batch, MK]
      Wf_x = Wf_x.transpose(0, 1)  # Shape: [batch, MK]
      # Compute L2 distances
    
      distances = -torch.norm(projected_flat - h_softmax_flat, p=2, dim=1)
      
      return projected_flat, distances
      
      #return projected_features, distances




    def process_batch(inputs, targets=None):
        final_output,B_weighted,h_softmax,reg_loss  = model(inputs)
        score= final_output
        reg_loss=-reg_loss
        #print(distances.shape)
        # distances is a tensor of shape [batch], convert to CPU and numpy if needed
        # Return distances along with h_softmax and score if you want
        return h_softmax.view(h_softmax.size(0), -1), score, reg_loss.cpu().numpy()
    # For ID data
    for split, in_loader in [('val', testloaderIn)]:
        cache_name = f"cache/{args.in_dataset}_{args.model_arch}_{split}_in_alllayers.npy"
        dist_cache_name = f"cache/{args.in_dataset}_{args.model_arch}_{split}_in_distances.npy"

        if FORCE_RUN or not os.path.exists(cache_name):
            feat_log = np.zeros((len(in_loader.dataset), sum(featdims)))
            score_log = np.zeros((len(in_loader.dataset), num_classes))
            label_log = np.zeros(len(in_loader.dataset))
            dist_log = np.zeros(len(in_loader.dataset))  # For OOD scores of ID data

            model.eval()
            n_test_acc = 0
            len_test_data = len(in_loader.dataset)
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(in_loader):
                    inputs, targets = inputs.to(args.device), targets.to(args.device)
                    start_ind = batch_idx * batch_size
                    end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

                    h_softmax_batch, score_batch, dist_batch = process_batch(inputs, targets)
                    feat_log[start_ind:end_ind, :] = h_softmax_batch.cpu().numpy()
                    label_log[start_ind:end_ind] = targets.cpu().numpy()
                    score_log[start_ind:end_ind] = score_batch.cpu().numpy()
                    dist_log[start_ind:end_ind] = dist_batch.squeeze()  # dist_batch is already a NumPy array if you returned it as such
                    # Compute accuracy for this batch
                    # Assuming score_batch are logits or probabilities for each class
                    y_hat = torch.max(score_batch, dim=1)[1]
                    n_test_acc += (y_hat == targets).sum().item()
                    if batch_idx % 100 == 0:
                        print(f"{batch_idx}/{len(in_loader)}")
            final_accuracy = n_test_acc / len_test_data
            print('Final test ID accuracy : {:.4f}'.format(final_accuracy))
            np.save(cache_name, np.array([feat_log.T, score_log.T, label_log], dtype=object))
            np.save(dist_cache_name, dist_log)
        else:
            # If already cached, just load
            feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
            feat_log, score_log = feat_log.T, score_log.T

    # For OOD data
    #d = ['SVHN', 'LSUN', 'iSUN']
    #d = ['SVHN', 'LSUN', 'iSUN', 'dtd', 'places365'] 
    d = ['SVHN', 'FashionMNIST','LSUN', 'iSUN', 'dtd', 'places365']
    for ood_dataset in d:
        out_loader = get_out_loader(ood_dataset, args)
        cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.model_arch}_out_alllayers.npy"
        dist_cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.model_arch}_out_distances.npy"

        if FORCE_RUN or not os.path.exists(cache_name):
            ood_feat_log = np.zeros((len(out_loader.dataset), sum(featdims)))
            ood_score_log = np.zeros((len(out_loader.dataset), num_classes))
            ood_dist_log = np.zeros(len(out_loader.dataset))  # OOD scores for OOD data

            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, _) in enumerate(out_loader):
                    inputs = inputs.to(args.device)
                    start_ind = batch_idx * batch_size
                    end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

                    h_softmax_batch, score_batch, dist_batch = process_batch(inputs)
                    
                    ood_feat_log[start_ind:end_ind, :] = h_softmax_batch.cpu().numpy()
                    ood_score_log[start_ind:end_ind] = score_batch.cpu().numpy()
                    ood_dist_log[start_ind:end_ind] = dist_batch.squeeze()  # dist_batch is already a NumPy array if you returned it as such

                    
                    if batch_idx % 100 == 0:
                        print(f"{batch_idx}/{len(out_loader)}")

            np.save(cache_name, np.array([ood_feat_log.T, ood_score_log.T], dtype=object))
            np.save(dist_cache_name, ood_dist_log)
        else:
            ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
            ood_feat_log, ood_score_log = ood_feat_log.T, ood_score_log.T

    print(time.time() - begin)

    # After extracting features and distances, call the OOD detection with confusion projection
    run_ood_detection_with_confusion_projection(args.in_dataset, args.model_arch, d, start, end)

   
