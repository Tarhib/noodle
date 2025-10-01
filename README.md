# Tackling the Noisy Elephant in the Room: Label Noise-robust Out-of-Distribution Detection via Loss Correction and Low-rank Decomposition

This is the soruce code of our proposed method **NOODLE** in paper **Tackling the Noisy Elephant in the Room: Label Noise-robust Out-of-Distribution Detection via Loss Correction and Low-rank Decomposition** .  
📄 [Paper Link (arXiv)](https://arxiv.org/abs/2509.06918)  

---

## Requirements

- Python 3.8+  
- PyTorch >= 1.10  
- torchvision  
- numpy  
- scikit-learn  
- faiss (optional, for nearest neighbor search)  

You can install dependencies via:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Dataset Preparation

Download in-distribution datasets and place them in `./data/`.  
For example, to download CIFAR-10:

```bash
cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
```

To download CIFAR-100 dataset:

```bash
cd data
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvzf cifar-100-python.tar.gz

```
**CIFAR-10N (Noisy Labels)**

The CIFAR-10N dataset provides human-annotated noisy labels.
Download from the official repository:
https://github.com/UCSC-REAL/cifar-10-100n

Extract and place inside ./data/

**CIFAR-100N (Noisy Labels)**

The CIFAR-100N dataset also provides human-annotated noisy labels.
Download from the same repository:
https://github.com/UCSC-REAL/cifar-10-100n

Extract and place inside ./data/cifar-100n/



#### OOD Datasets

For OOD detection experiments, you can download the commonly used datasets as follows:

- **SVHN**:  
  http://ufldl.stanford.edu/housenumbers/

- **LSUN (Crop & Resize)**:  
  http://www.yf.io/p/lsun  

- **iSUN**:  
  https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz?dl=0  

- **Textures (DTD)**:  
  https://www.robots.ox.ac.uk/~vgg/data/dtd/

- **Places365**:  
  http://places2.csail.mit.edu/download.html  

Example (download and extract iSUN):

```bash
cd data
wget https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz?dl=0 -O iSUN.tar.gz
tar -xvzf iSUN.tar.gz
```
For FashionMNIST, it will auto download using torch library.
Make sure all OOD datasets follow this structure:

```
./ood_data/
    SVHN/
    FashionMNIST
    LSUN/
    iSUN/
    dtd/
    Places365/
```

### 2. Training

We have different training script for each dataset.
For CIFAR-10 run below one:

```bash
python train_densenet.py --id CIFAR-10 --bs 64 --r 1
```
For Animal-10N run below one:

```bash
python train_densenet_animal.py --id CIFAR-100 --bs 128 --r 1 
```


For CIFAR-100 run below one:

```bash
python train_densenet_cifar100.py --id CIFAR-100 --bs 128 --r 1 
```



### 3. Evaluation / OOD Detection

Run evaluation with in-dataset as **CIFAR-10N** with following command:

```bash
python test_cifar.py --in-dataset CIFAR-10 --model_arch densenet --bs 64
```

If in-dataset is  **CIFAR-100N**
```bash
python test_cifar.py --in-dataset CIFAR-100 --model_arch densenet --bs 128  --K 100
```

If in-dataset is  **Animal-10N**
```bash
python test_cifar.py --in-dataset Animal10n --model_arch densenet --bs 64 --K 10
```


---

<!-- ## Reference
you find this code useful, please cite: -->

<!-- ```bibtex
@inproceedings{yourcitation2025,
  title={Your Paper Title},
  author={Your Name and Collaborators},
  booktitle={Proceedings of ...},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License.  
See the LICENSE file for details. -->
