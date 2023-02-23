import sys
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import argparse
import logging
from metrics import evaluate
from utils import masked_mae, StandardScaler, getDayTimestamp
from MegaCRNx import MegaCRN

def getXSYS(data, mode):
    train_num = int(data.shape[0] * opt.trainval_ratio)
    XS, YS = [], []
    if mode == 'train':    
        for i in range(train_num - opt.seq_len - opt.his_len + 1):
            x = data[i:i+opt.his_len, ...]
            y = data[i+opt.his_len:i+opt.his_len+opt.seq_len, ...]
            XS.append(x), YS.append(y)
    elif mode == 'test':
        for i in range(train_num - opt.his_len,  data.shape[0] - opt.seq_len - opt.his_len + 1):
            x = data[i:i+opt.his_len, ...]
            y = data[i+opt.his_len:i+opt.his_len+opt.seq_len, ...]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    return XS, YS

def getXSYSTIME(data, data_time, mode):
    train_num = int(data.shape[0] * opt.trainval_ratio)
    XS, YS, YS_TIME = [], [], []
    if mode == 'train':    
        for i in range(train_num - opt.seq_len - opt.his_len + 1):
            x = data[i:i+opt.his_len, ...]
            y = data[i+opt.his_len:i+opt.his_len+opt.seq_len, ...]
            t = data_time[i+opt.his_len:i+opt.his_len+opt.seq_len, ...]
            XS.append(x), YS.append(y), YS_TIME.append(t)
    elif mode == 'test':
        for i in range(train_num - opt.his_len,  data.shape[0] - opt.seq_len - opt.his_len + 1):
            x = data[i:i+opt.his_len, ...]
            y = data[i+opt.his_len:i+opt.his_len+opt.seq_len, ...]
            t = data_time[i+opt.his_len:i+opt.his_len+opt.seq_len, ...]
            XS.append(x), YS.append(y), YS_TIME.append(t)
    XS, YS, YS_TIME = np.array(XS), np.array(YS), np.array(YS_TIME)
    XS, YS, YS_TIME = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis], YS_TIME[:, :, :, np.newaxis]
    return XS, YS, YS_TIME

def print_params(model):
    # print trainable params
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    logger.info(f'In total: {param_count} trainable parameters. \n')
    return

def getModel(mode):
    model = MegaCRN(num_nodes=num_variable, input_dim=opt.channelin, output_dim=opt.channelout, horizon=opt.seq_len, 
                        rnn_units=opt.hiddenunits, num_layers=opt.num_layers, mem_num=opt.mem_num, mem_dim=opt.mem_dim, 
                        memory_type=opt.memory, meta_type=opt.meta, decoder_type=opt.decoder).to(device)
    if mode == 'train':
        summary(model, [(opt.his_len, num_variable, opt.channelin), (opt.seq_len, num_variable, opt.channelout)], device=device)   
        print_params(model)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    return model

def evaluateModel(model, data_iter):
    if opt.loss == 'MAE': 
        criterion = nn.L1Loss()
        separate_loss = nn.TripletMarginLoss(margin=1.0)
        compact_loss = nn.MSELoss()
    if opt.loss == 'MaskMAE':
        criterion = masked_mae
        separate_loss = nn.TripletMarginLoss(margin=1.0)
        compact_loss = nn.MSELoss()
        
    model.eval()
    loss_sum, n, YS_pred = 0.0, 0, []
    loss_sum1, loss_sum2, loss_sum3 = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y, y_cov in data_iter:
            y_pred, h_att, query, pos, neg = model(x, y_cov)
            y_pred = scaler.inverse_transform(y_pred)
            loss1 = criterion(y_pred, y)
            loss2 = separate_loss(query, pos.detach(), neg.detach())
            loss3 = compact_loss(query, pos.detach())
            loss = loss1 + opt.lamb * loss2 + opt.lamb1 * loss3
            loss_sum += loss.item() * y.shape[0]
            loss_sum1 += loss1.item() * y.shape[0]
            loss_sum2 += loss2.item() * y.shape[0]
            loss_sum3 += loss3.item() * y.shape[0]
            n += y.shape[0]
            YS_pred.append(y_pred.cpu().numpy())     
    loss, loss1, loss2, loss3 = loss_sum / n, loss_sum1 / n, loss_sum2 / n, loss_sum3 / n
    YS_pred = np.vstack(YS_pred)
    return loss, loss1, loss2, loss3, YS_pred

def trainModel(name, mode, XS, YS, YCov):
    model = getModel(mode)
    
    XS = scaler.transform(XS)
    XS_torch, YS_torch, YCov_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device), torch.Tensor(YCov).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch, YCov_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - opt.val_ratio))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, opt.batch_size, shuffle=False)
    val_iter = torch.utils.data.DataLoader(val_data, opt.batch_size, shuffle=False)
    trainval_iter = torch.utils.data.DataLoader(trainval_data, opt.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    if opt.loss == 'MAE': 
        criterion = nn.L1Loss()
        separate_loss = nn.TripletMarginLoss(margin=1.0)
        compact_loss = nn.MSELoss()
    if opt.loss == 'MaskMAE':
        criterion = masked_mae
        separate_loss = nn.TripletMarginLoss(margin=1.0)
        compact_loss = nn.MSELoss()
        
    min_val_loss = np.inf
    wait = 0   
    for epoch in range(opt.epoch):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        loss_sum1, loss_sum2, loss_sum3 = 0.0, 0.0, 0.0
        model.train()
        for x, y, ycov in train_iter:
            optimizer.zero_grad()
            y_pred, h_att, query, pos, neg = model(x, ycov)
            y_pred = scaler.inverse_transform(y_pred)
            loss1 = criterion(y_pred, y)
            loss2 = separate_loss(query, pos.detach(), neg.detach())
            loss3 = compact_loss(query, pos.detach())
            loss = loss1 + opt.lamb * loss2 + opt.lamb1 * loss3
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            loss_sum1 += loss1.item() * y.shape[0]
            loss_sum2 += loss2.item() * y.shape[0]
            loss_sum3 += loss3.item() * y.shape[0]
            n += y.shape[0]
        train_loss, train_loss1, train_loss2, train_loss3 = loss_sum / n, loss_sum1 / n, loss_sum2 / n, loss_sum3 / n
        val_loss, val_loss1, val_loss2, val_loss3, _ = evaluateModel(model, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
        else:
            wait += 1
            if wait == opt.patience:
                logger.info('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        logger.info("epoch", epoch, "time used:", epoch_time, "seconds", 
                    "train loss:", '%.6f %.6f %.6f %.6f' % (train_loss, train_loss1, train_loss2, train_loss3),
                    "validation loss:", '%.6f %.6f %.6f %.6f' % (val_loss, val_loss1, val_loss2, val_loss3))
        with open(epochlog_path, 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.6f, %s, %.6f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
    
    loss, loss1, loss2, loss3, YS_pred = evaluateModel(model, trainval_iter)
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred)
    MSE, RMSE, MAE, MAPE = evaluate(YS, YS_pred)
    logger.info("%s, %s, trainval loss, loss1, loss2, loss3, %.6f, %.6f, %.6f, %.6f" % (name, mode, loss, loss1, loss2, loss3))
    logger.info("%s, %s, valid loss, loss1, loss2, loss3, %.6f, %.6f, %.6f, %.6f" % (name, mode, val_loss, val_loss1, val_loss2, val_loss3))
    logger.info("%s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f" % (name, mode, MSE, RMSE, MAE, MAPE))
    
def testModel(name, mode, XS, YS, YCov):
    model = getModel(mode)
    model.load_state_dict(torch.load(modelpt_path))
    
    XS = scaler.transform(XS)
    XS_torch, YS_torch, YCov_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device), torch.Tensor(YCov).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch, YCov_torch)
    test_iter = torch.utils.data.DataLoader(test_data, opt.batch_size, shuffle=False)
    loss, loss1, loss2, loss3, YS_pred = evaluateModel(model, test_iter)
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred)
    np.save(path + f'/{name}_prediction.npy', YS_pred)
    np.save(path + f'/{name}_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = evaluate(YS, YS_pred)
    logger.info("%s, %s, test loss, loss1, loss2, loss3, %.6f, %.6f, %.6f, %.6f" % (name, mode, loss, loss1, loss2, loss3))
    logger.info("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f" % (name, mode, MSE, RMSE, MAE, MAPE))
    with open(score_path, 'a') as f:
        f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
        for i in range(opt.seq_len):
            MSE, RMSE, MAE, MAPE = evaluate(YS[:, i, :], YS_pred[:, i, :])
            logger.info("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
            f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        
#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY'], default='METRLA', help='which dataset to run')
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the ratio of training and validation data among the total')
parser.add_argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--seq_len', type=int, default=12, help='sequence length of prediction')
parser.add_argument('--his_len', type=int, default=12, help='sequence length of historical observation')
parser.add_argument('--channelin', type=int, default=1, help='number of input channel')
parser.add_argument('--channelout', type=int, default=1, help='number of output channel')
parser.add_argument("--loss", type=str, default='MaskMAE', help="MAE, MSE, MaskMAE")
parser.add_argument("--epoch", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--patience", type=float, default=10, help="patience used for early stop")
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--hiddenunits', type=int, default=32, help='number of hidden units')
parser.add_argument('--mem_num', type=int, default=10, help='number of meta-nodes/prototypes')
parser.add_argument('--mem_dim', type=int, default=32, help='dimension of meta-nodes/prototypes')
parser.add_argument("--memory", type=eval, choices=[True, False], default='True', help="whether to use memory: True or False")
parser.add_argument("--meta", type=eval, choices=[True, False], default='True', help="whether to use meta-graph: True or False")
parser.add_argument("--decoder", type=str, choices=['sequence', 'stepwise'], default='stepwise', help="decoder type: sequence or stepwise")
parser.add_argument('--lamb', type=float, default=0.01, help='lamb value for separate loss')
parser.add_argument('--lamb1', type=float, default=0.01, help='lamb1 value for compact loss')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--seed', type=int, default=100, help='random seed.')
opt = parser.parse_args()

model_name = 'MegaCRNx'
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'../save/{opt.dataset}_{model_name}_{timestring}'
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
score_path = f'{path}/{model_name}_{timestring}_scores.txt'
epochlog_path = f'{path}/{model_name}_{timestring}_epochlog.txt'
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
if not os.path.exists(path): os.makedirs(path)
shutil.copy2(sys.argv[0], path)
shutil.copy2(f'{model_name}.py', path)
shutil.copy2('utils.py', path)
shutil.copy2('metrics.py', path)
    
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
class MyFormatter(logging.Formatter):
    def format(self, record):
        spliter = ' '
        record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
        record.args = tuple() # set empty to args
        return super().format(record)
formatter = MyFormatter()
handler = logging.FileHandler(logging_path, mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

logger.info('model_name', model_name)
logger.info('dataset_name', opt.dataset)
logger.info('memory_type', opt.memory)
logger.info('meta_type', opt.meta)
logger.info('decoder_type', opt.decoder)
logger.info('loss', opt.loss)
logger.info('separate loss lamb', opt.lamb)
logger.info('compact loss lamb1', opt.lamb1)
logger.info('batch_size', opt.batch_size)
logger.info('mem_num', opt.mem_num)
logger.info('mem_dim', opt.mem_dim)
logger.info('rnn_units', opt.hiddenunits)
logger.info('num_layers', opt.num_layers)
logger.info('channnel_in', opt.channelin)
logger.info('channnel_out', opt.channelout)

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
device = torch.device("cuda:{}".format(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
# Please comment the following three lines for running experiments multiple times.
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(opt.seed)
#####################################################################################################

if opt.dataset == 'METRLA':
    num_variable = 207
    data_path = f'../{opt.dataset}/metr-la.h5'
elif opt.dataset == 'PEMSBAY':
    num_variable = 325
    data_path = f'../{opt.dataset}/pems-bay.h5'
else:
    pass # including more datasets in the future    

data = pd.read_hdf(data_path).values
data_time = getDayTimestamp(pd.read_hdf(data_path))
mean = np.mean(data[:int(data.shape[0]*opt.trainval_ratio)])
std = np.std(data[:int(data.shape[0]*opt.trainval_ratio)])
scaler = StandardScaler(mean, std)

def main():
    logger.info(opt.dataset, 'training started', time.ctime())
    trainXS, trainYS, trainYCov = getXSYSTIME(data, data_time, 'train')
    logger.info('TRAIN XS.shape YS.shape, YCov.shape', trainXS.shape, trainYS.shape, trainYCov.shape)
    trainModel(model_name, 'train', trainXS, trainYS, trainYCov)
    logger.info(opt.dataset, 'training ended', time.ctime())
    logger.info('=' * 90)
    
    logger.info(opt.dataset, 'testing started', time.ctime())
    testXS, testYS, testYCov = getXSYSTIME(data, data_time, 'test')
    logger.info('TEST XS.shape, YS.shape, YCov.shape', testXS.shape, testYS.shape, testYCov.shape)
    testModel(model_name, 'test', testXS, testYS, testYCov)
    logger.info(opt.dataset, 'testing ended', time.ctime())
    logger.info('=' * 90)

if __name__ == '__main__':
    main()
