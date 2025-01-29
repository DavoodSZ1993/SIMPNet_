#!/usr/bin/env python3

import torch 
import numpy as np 
import random
from torch_geometric.data import Batch

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def custom_collate_fn(batch):

    # Unzip the batch
    batch1, batch2 = zip(*batch)

    # Handle the PyTorch Geometric data
    batch1 = Batch.from_data_list(batch1)

    # Handle the standard PyTorch data
    batch2 = torch.utils.data.dataloader.default_collate(batch2)

    return batch1, batch2

def z_score(X, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu)/ \sigma$,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: torch array, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return torch array, z-score normalized array
    '''
    eps = 1e-3
    return (x - mean)/(std+eps)

def un_z_score(x_normed, mean, std):
    '''
    Undo the Z-score calculation.
    '''
    return x_normed * std + mean

def MAPE(v, v_):
    '''
    Mean absolute percentage error given as a % (e.g., 99 -> 99%)
    '''
    return torch.mean(torch.abs(v_ - v)/ (v + 1e-15)*100)

def RSME(v, v_):
    '''
    Mean suqarred error
    '''
    return torch.sqrt(torch.mean((v_ - v)**2))

def MAE(v, v_):
    '''
    Mean absolute error.
    '''
    return torch.mean(torch.abs(v_ - v))



