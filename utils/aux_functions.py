import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as DataUtils
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import sys

# %%
# Readymade data loading function
DATA_ROOT='./data/'

def get_data_loaders(dataset, batch_size=64, val=False):
  
  Transform = transforms.Compose([transforms.ToTensor()])
  if dataset == 'MNIST':
    train_set = datasets.MNIST(root=DATA_ROOT, download=True, train=True, transform=Transform)
    val_set = datasets.MNIST(root=DATA_ROOT, download=True, train=True, transform=Transform)
    test_set = datasets.MNIST(root=DATA_ROOT, download=True, train=False, transform=Transform)
    n_train = 50000
  elif dataset == 'FMNIST':
    train_set = datasets.FashionMNIST(root=DATA_ROOT, download=True, train=True, transform=Transform)
    val_set = datasets.FashionMNIST(root=DATA_ROOT, download=True, train=True, transform=Transform)
    test_set = datasets.FashionMNIST(root=DATA_ROOT, download=True, train=False, transform=Transform)
    n_train = 50000
  elif dataset == 'CIFAR10':
    train_set = datasets.CIFAR10(root=DATA_ROOT, download=True, train=True, transform=Transform)
    val_set = datasets.CIFAR10(root=DATA_ROOT, download=True, train=True, transform=Transform)
    test_set = datasets.CIFAR10(root=DATA_ROOT, download=True, train=False, transform=Transform)
    n_train = 40000
  elif dataset == 'CIFAR100':
    train_set = datasets.CIFAR100(root=DATA_ROOT, download=True, train=True, transform=Transform)
    val_set = datasets.CIFAR100(root=DATA_ROOT, download=True, train=True, transform=Transform)
    test_set = datasets.CIFAR100(root=DATA_ROOT, download=True, train=False, transform=Transform)
    n_train = 40000
  else:
    print('Dataset is not defined. Exiting.')
    exit()
    
  if val:
    indices = np.arange(0, len(train_set))
    np.random.shuffle(indices)
    trainSampler = SubsetRandomSampler(indices[:n_train])
    valSampler = SubsetRandomSampler(indices[n_train:])
    testSampler = SubsetRandomSampler(np.arange(0, len(test_set)))
    train_loader = DataUtils.DataLoader(train_set, batch_size=batch_size, sampler=trainSampler)
    val_loader = DataUtils.DataLoader(val_set, batch_size=batch_size, sampler=valSampler)
    test_loader = DataUtils.DataLoader(test_set, batch_size=batch_size, sampler=testSampler)

  else:
    train_loader = DataUtils.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = None
    test_loader = DataUtils.DataLoader(test_set, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader, test_loader

# %%
# Utility progress bar function
def progress(curr, total, suffix=''):
  bar_len = 48
  filled = int(round(bar_len * curr / float(total)))
  if filled == 0:
    filled = 1
  bar = '=' * (filled - 1) + '>' + '-' * (bar_len - filled)
  print('\r[%s] %s' % (bar, suffix), end="")
  sys.stdout.flush()
  if curr == total:
    bar = bar_len * '='