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
from configparser import ConfigParser
import logging
from metrics import evaluate
from utils import *
from MegaCRN import MegaCRN
from sklearn.preprocessing import StandardScaler # For EXPY-TKY, we use sklearn 

def getModel(mode):
    model = MegaCRN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, horizon=args.horizon, 
                    rnn_units=args.rnn_units, num_layers=args.num_rnn_layers, mem_num=args.mem_num, mem_dim=args.mem_dim, 
                    cheb_k=args.max_diffusion_step, cl_decay_steps=args.cl_decay_steps, use_curriculum_learning=args.use_curriculum_learning).to(device)
    if mode == 'train':
        summary(model, [(args.seq_len, args.num_nodes, args.input_dim), (args.horizon, args.num_nodes, args.output_dim)], device=device)   
        print_params(model)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    return model

def evaluateModel(model, data_iter):
    if args.loss == 'MAE': criterion = nn.L1Loss()
    separate_loss = nn.TripletMarginLoss(margin=1.0)
    compact_loss = nn.MSELoss()
        
    model.eval()
    loss_sum, n, YS_pred = 0.0, 0, []
    loss_sum1, loss_sum2, loss_sum3 = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y, y_cov in data_iter:
            y_pred, h_att, query, pos, neg = model(x, y_cov)
            loss1 = criterion(y_pred, y)
            loss2 = separate_loss(query, pos.detach(), neg.detach())
            loss3 = compact_loss(query, pos.detach())
            loss = loss1 + args.lamb * loss2 + args.lamb1 * loss3
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
    
    XS_torch, YS_torch, YCov_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device), torch.Tensor(YCov).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch, YCov_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - args.val_ratio))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    
    if args.loss == 'MAE': criterion = nn.L1Loss()
    separate_loss = nn.TripletMarginLoss(margin=1.0)
    compact_loss = nn.MSELoss()
        
    min_val_loss = np.inf
    wait = 0
    batches_seen = 0
    for epoch in range(args.epochs):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        loss_sum1, loss_sum2, loss_sum3 = 0.0, 0.0, 0.0
        model.train()
        for x, y, ycov in train_iter:
            optimizer.zero_grad()
            y_pred, h_att, query, pos, neg = model(x, ycov, y, batches_seen)
            loss1 = criterion(y_pred, y)
            loss2 = separate_loss(query, pos.detach(), neg.detach())
            loss3 = compact_loss(query, pos.detach())
            loss = loss1 + args.lamb * loss2 + args.lamb1 * loss3
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            loss_sum1 += loss1.item() * y.shape[0]
            loss_sum2 += loss2.item() * y.shape[0]
            loss_sum3 += loss3.item() * y.shape[0]
            n += y.shape[0]
            batches_seen += 1
        lr_scheduler.step()
        train_loss, train_loss1, train_loss2, train_loss3 = loss_sum / n, loss_sum1 / n, loss_sum2 / n, loss_sum3 / n
        val_loss, val_loss1, val_loss2, val_loss3, _ = evaluateModel(model, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
        else:
            wait += 1
            if wait == args.patience:
                logger.info('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        logger.info("epoch", epoch, "time used:", epoch_time, "seconds", 
                    "train loss:", '%.6f %.6f %.6f %.6f' % (train_loss, train_loss1, train_loss2, train_loss3),
                    "validation loss:", '%.6f %.6f %.6f %.6f' % (val_loss, val_loss1, val_loss2, val_loss3))
        with open(epochlog_path, 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.6f, %s, %.6f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "valid loss:", val_loss))
    
def testModel(name, mode, XS, YS, YCov):
    model = getModel(mode)
    model.load_state_dict(torch.load(modelpt_path))
    
    XS_torch, YS_torch, YCov_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device), torch.Tensor(YCov).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch, YCov_torch)
    test_iter = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False)
    loss, loss1, loss2, loss3, YS_pred = evaluateModel(model, test_iter)
    
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred) 
    YS, YS_pred = YS.reshape(-1, YS.shape[-1]), YS_pred.reshape(-1, YS_pred.shape[-1])
    YS, YS_pred = scaler.inverse_transform(YS), scaler.inverse_transform(YS_pred)
    YS, YS_pred = YS.reshape(-1, args.horizon, YS.shape[-1]), YS_pred.reshape(-1, args.horizon, YS_pred.shape[-1])
    # np.save(path + f'/{name}_prediction.npy', YS_pred)
    # np.save(path + f'/{name}_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = evaluate(YS, YS_pred)
    
    logger.info("%s, %s, test loss, loss1, loss2, loss3, %.6f, %.6f, %.6f, %.6f" % (name, mode, loss, loss1, loss2, loss3))
    logger.info("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f" % (name, mode, MSE, RMSE, MAE, MAPE))
    with open(score_path, 'a') as f:
        f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
        for i in range(args.horizon):
            MSE, RMSE, MAE, MAPE = evaluate(YS[:, i, :], YS_pred[:, i, :])
            logger.info("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
            f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.6f, %.6f, %.6f, %.6f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))

#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['EXPYTKY', 'EXPYTKY*'], default='EXPYTKY', help='which dataset to run')
parser.add_argument('--month', type=str, default='202112', help='which experiment setting (month) to run as testing data')
parser.add_argument('--val_ratio', type=float, default=0.25, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--num_nodes', type=int, default=1843, help='num_nodes')
parser.add_argument('--seq_len', type=int, default=6, help='input sequence length')
parser.add_argument('--horizon', type=int, default=6, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
parser.add_argument('--max_diffusion_step', type=int, default=3, help='max diffusion step or Cheb K')
parser.add_argument('--num_rnn_layers', type=int, default=1, help='number of rnn layers')
parser.add_argument('--rnn_units', type=int, default=32, help='number of rnn units')
parser.add_argument('--mem_num', type=int, default=10, help='number of meta-nodes/prototypes')
parser.add_argument('--mem_dim', type=int, default=32, help='dimension of meta-nodes/prototypes')
parser.add_argument("--loss", type=str, default='MAE', help="MAE, MSE, MaskMAE")
parser.add_argument('--lamb', type=float, default=0.01, help='lamb value for separate loss')
parser.add_argument('--lamb1', type=float, default=0.0, help='lamb1 value for compact loss')
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="base learning rate")
parser.add_argument("--steps", type=eval, default=[200], help="steps")
parser.add_argument("--patience", type=int, default=10, help="patience used for early stop")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning")
parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
# parser.add_argument('--seed', type=int, default=100, help='random seed')
args = parser.parse_args()

config = ConfigParser()
config.read('params.txt', encoding='UTF-8')
train_month = eval(config[args.month]['train_month'])
test_month = eval(config[args.month]['test_month'])
traffic_path = config[args.month]['traffic_path']
subroad_path = config[args.dataset]['subroad_path']
road_path = config['common']['road_path']
#adj_path = config['common']['adjdis_path'] 
adj_path = config['common']['adj01_path']
num_variable = len(np.loadtxt(subroad_path).astype(int))
N_link = config.getint('common', 'N_link')
args.num_nodes = num_variable

model_name = 'MegaCRN'
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'../save/{args.dataset}_{model_name}_{timestring}'
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
score_path = f'{path}/{model_name}_{timestring}_scores.txt'
epochlog_path = f'{path}/{model_name}_{timestring}_epochlog.txt'
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
if not os.path.exists(path): os.makedirs(path)
shutil.copy2(sys.argv[0], path)
shutil.copy2(f'{model_name}.py', path)
shutil.copy2('utils.py', path)
shutil.copy2('metrics.py', path)
shutil.copy2('params.txt', path)

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

logger.info('model', model_name)
logger.info('dataset', args.dataset)
logger.info('val_ratio', args.val_ratio)
logger.info('num_nodes', args.num_nodes)
logger.info('seq_len', args.seq_len)
logger.info('horizon', args.horizon)
logger.info('input_dim', args.input_dim)
logger.info('output_dim', args.output_dim)
logger.info('num_rnn_layers', args.num_rnn_layers)
logger.info('rnn_units', args.rnn_units)
logger.info('max_diffusion_step', args.max_diffusion_step)
logger.info('mem_num', args.mem_num)
logger.info('mem_dim', args.mem_dim)
logger.info('loss', args.loss)
logger.info('separate loss lamb', args.lamb)
logger.info('compact loss lamb1', args.lamb1)
logger.info('epochs', args.epochs)
logger.info('batch_size', args.batch_size)
logger.info('lr', args.lr)
logger.info('steps', args.steps)
logger.info('patience', args.patience)
logger.info('use_curriculum_learning', args.use_curriculum_learning)

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
# Please comment the following three lines for running experiments multiple times.
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
#####################################################################################################

scaler = StandardScaler()

def main():
    train_data = [get_data(config[month]['traffic_path'], N_link, subroad_path, ['speed']) for month in train_month]
    test_data = [get_data(config[month]['traffic_path'], N_link, subroad_path, ['speed']) for month in test_month]
    train_time = [get_time(config[month]['traffic_path'], N_link, subroad_path) for month in train_month]
    test_time = [get_time(config[month]['traffic_path'], N_link, subroad_path) for month in test_month]
        
    speed_data = [data[:,:,0] for data in train_data+test_data] # better to only use train_data
    speed_data = np.vstack(speed_data)    
    scaler.fit(speed_data)
    for data in train_data + test_data:
        data[:,:,0] = scaler.transform(data[:,:,0])
    
    logger.info(args.dataset, args.month, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(train_data, args.seq_len, args.horizon)
    _, trainYCov = getXSYS(train_time, args.seq_len, args.horizon)
    logger.info('TRAIN XS.shape YS.shape, YCov.shape', trainXS.shape, trainYS.shape, trainYCov.shape)
    trainModel(model_name, 'train', trainXS, trainYS, trainYCov)
    logger.info(args.dataset, args.month, 'training ended', time.ctime())
    logger.info('=' * 90)
    
    logger.info(args.dataset, args.month, 'testing started', time.ctime())
    testXS, testYS = getXSYS(test_data, args.seq_len, args.horizon)
    _, testYCov = getXSYS(test_time, args.seq_len, args.horizon)
    logger.info('TEST XS.shape, YS.shape, YCov.shape', testXS.shape, testYS.shape, testYCov.shape)
    testModel(model_name, 'test', testXS, testYS, testYCov)
    logger.info(args.dataset, args.month, 'testing ended', time.ctime())
    logger.info('=' * 90)

if __name__ == '__main__':
    main()