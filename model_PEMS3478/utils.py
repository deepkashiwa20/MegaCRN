import pickle
import torch
import numpy as np
import pandas as pd

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class StandardScalerLocal():
    def __init__(self, init_data, device):
        self.mean = np.mean(init_data, axis=0)
        self.std = np.std(init_data, axis=0)
        self.mean_torch = torch.Tensor(self.mean)[:, None].to(device)
        self.std_torch = torch.Tensor(self.std)[:, None].to(device)
    
    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std_torch) + self.mean_torch
    
def getTimestamp(data):
    num_samples, num_nodes = data.shape
    time_ind = (data.index.values - data.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [num_nodes,1]).transpose((1, 0))
    return time_in_day

def getDayTimestamp(data):
    # 288 timeslots each day for dataset has 5 minutes time interval.
    df = pd.DataFrame({'timestamp':data.index.values})
    df['weekdaytime'] = df['timestamp'].dt.weekday * 288 + (df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute)//5
    df['weekdaytime'] = df['weekdaytime'] / df['weekdaytime'].max()
    num_samples, num_nodes = data.shape
    time_ind = df['weekdaytime'].values
    time_ind_node = np.tile(time_ind, [num_nodes,1]).transpose((1, 0))
    return time_ind_node

def getDayTimestamp_(start, end, freq, num_nodes):
    # 288 timeslots each day for dataset has 5 minutes time interval.
    df = pd.DataFrame({'timestamp':pd.date_range(start=start, end=end, freq=freq)})
    df['weekdaytime'] = df['timestamp'].dt.weekday * 288 + (df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute)//5
    df['weekdaytime'] = df['weekdaytime'] / df['weekdaytime'].max()
    time_ind = df['weekdaytime'].values
    time_ind_node = np.tile(time_ind, [num_nodes, 1]).transpose((1, 0))
    return time_ind_node

def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

# DCRNN
def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'\n In total: {param_count} trainable parameters. \n')
    return