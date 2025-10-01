from pickle import FALSE
import torch
import os
import numpy as np
import torch.nn.functional as F
import time
import torchvision
import torchvision.transforms as transforms
import models.densenet as dn
import numpy as np
import time
import argparse
#from feat_extract_encoded_animal import feat_extract
from feat_extract_encoded import feat_extract
#from feat_maha import feat_extract
#from feat_extract_animal_ss import feat_extract

#from feat_extract_maha import feat_extract
#from feat_extract_mixed import feat_extract
#from feat_extract_subspace import feat_extract_subspace
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='SNN testing with CIFAR benchmark')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--model_arch', default='densenet', type=str, help='model architecture ([densenet, resnet50]')
parser.add_argument('--bs', default = 200, type = int, help='Batch size')
parser.add_argument('--M',type=int,help='No of annotators',default=5)
parser.add_argument('--K',type=int,help='No of classes',default=10)
parser.add_argument('--layers', default=100, type=int, help='total number of layers (default: 100)')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.1)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.set_defaults(argument=True)

args = parser.parse_args()
args.device = device


if __name__ == '__main__':
    #feat_extract(args, use_mahalanobis=True)
    feat_extract(args)