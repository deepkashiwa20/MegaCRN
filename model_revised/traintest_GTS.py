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
from utils import StandardScaler, DataLoader
from GTS import GTSModel, masked_mae_loss, masked_mape_loss, masked_mse_loss, masked_rmse_loss, count_parameters

def print_model(model):
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    logger.info(f'In total: {param_count} trainable parameters.')
    return

def get_model():
    model = GTSModel(gpu=args.gpu, 
                     temperature=args.temperature, 
                     cl_decay_steps=args.cl_decay_steps, 
                     filter_type=args.filter_type, 
                     max_diffusion_step=args.max_diffusion_step,
                     num_nodes=args.num_nodes,
                     num_rnn_layers=args.num_rnn_layers,
                     rnn_units=args.rnn_units,
                     input_dim=args.input_dim,
                     output_dim=args.output_dim,
                     horizon=args.horizon,
                     seq_len=args.seq_len,
                     use_curriculum_learning=args.use_curriculum_learning, 
                     dim_fc=args.dim_fc).to(device)
    return model

def init_model(model):
    with torch.no_grad():
        model = model.eval()
        val_iter =  data['val_loader'].get_iterator()
        for x, y in val_iter:
            x, y = prepare_x_y(x, y)
            output = model(x, train_feas)
            break
        return model

def prepare_x_y(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :return1: x shape (seq_len, batch_size, num_sensor, input_dim)
              y shape (horizon, batch_size, num_sensor, input_dim)
    :return2: x: shape (seq_len, batch_size, num_sensor * input_dim)
              y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x = x.permute(1, 0, 2, 3)
    y = y.permute(1, 0, 2, 3)
    batch_size = x.size(1)
    x = x.view(args.seq_len, batch_size, args.num_nodes * args.input_dim)
    y = y[..., :args.output_dim].view(args.horizon, batch_size, args.num_nodes * args.output_dim)
    return x.to(device), y.to(device)

def compute_loss(y_true, y_predicted):
    y_true = scaler.inverse_transform(y_true)
    y_predicted = scaler.inverse_transform(y_predicted)
    return masked_mae_loss(y_predicted, y_true) # masked_mae or masked_mae_loss
    
def evaluate_metric(ys_true, ys_pred):
    maes, mapes, mses = [], [], []
    l_3, m_3, r_3 = [], [], []
    l_6, m_6, r_6 = [], [], []
    l_12, m_12, r_12 = [], [], []
    for y_true, y_pred in zip(ys_true, ys_pred):
        # Followed the DCRNN TensorFlow Implementation
        maes.append(masked_mae_loss(y_pred, y_true).item())
        mapes.append(masked_mape_loss(y_pred, y_true).item())
        mses.append(masked_mse_loss(y_pred, y_true).item())
        l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
        m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
        r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
        l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
        m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
        r_6.append(masked_mse_loss(y_pred[5:6], y_true[5:6]).item())
        l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
        m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
        r_12.append(masked_mse_loss(y_pred[11:12], y_true[11:12]).item())
    mean_mae, mean_mape, mean_rmse = np.mean(maes), np.mean(mapes), np.sqrt(np.mean(mses))
    l_3, m_3, r_3 = np.mean(l_3), np.mean(m_3), np.sqrt(np.mean(r_3))
    l_6, m_6, r_6 = np.mean(l_6), np.mean(m_6), np.sqrt(np.mean(r_6))
    l_12, m_12, r_12 = np.mean(l_12), np.mean(m_12), np.sqrt(np.mean(r_12))
    logger.info('Horizon overall: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_mae, mean_mape, mean_rmse))
    logger.info('Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_3, m_3, r_3))
    logger.info('Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_6, m_6, r_6))
    logger.info('Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_12, m_12, r_12))
    return mean_mae, mean_mape, mean_rmse, l_3, m_3, r_3, l_6, m_6, r_6, l_12, m_12, r_12
    
def evaluate(model, mode):
    with torch.no_grad():
        model = model.eval()
        data_iter =  data[f'{mode}_loader'].get_iterator()
        losses = []
        ys_true, ys_pred = [], []
        for x, y in data_iter:
            x, y = prepare_x_y(x, y)
            output, mid_output = model(x, train_feas)
            loss_1 = compute_loss(y, output)
            pred = torch.sigmoid(mid_output.view(mid_output.shape[0] * mid_output.shape[1]))
            true_label = adj_mx.view(mid_output.shape[0] * mid_output.shape[1])
            bce_loss = torch.nn.BCELoss()
            loss_g = bce_loss(pred, true_label)
            loss = loss_1 + loss_g
            losses.append((loss_1.item()+loss_g.item()))
            
            y_true = scaler.inverse_transform(y)
            y_pred = scaler.inverse_transform(output)
            ys_true.append(y_true)
            ys_pred.append(y_pred)

        mean_loss = np.mean(losses)
        return mean_loss, ys_true, ys_pred

def traintest_model():  
    model = get_model()
    model = init_model(model)
    print_model(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, eps=args.epsilon)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    min_val_loss = float('inf')
    wait = 0
    batches_seen = 0
    for epoch_num in range(args.epochs):
        model = model.train()
        train_iter = data['train_loader'].get_iterator()
        losses = []
        start_time = time.time()

        for x, y in train_iter:
            optimizer.zero_grad()
            x, y = prepare_x_y(x, y)
            output, mid_output = model(x, train_feas, y, batches_seen)
            loss_1 = compute_loss(y, output)
            pred = mid_output.view(mid_output.shape[0] * mid_output.shape[1])
            true_label = adj_mx.view(mid_output.shape[0] * mid_output.shape[1])
            bce_loss = torch.nn.BCELoss()
            loss_g = bce_loss(pred, true_label)
            loss = loss_1 + loss_g
            losses.append((loss_1.item()+loss_g.item()))
            batches_seen += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # gradient clipping - this does it in place
            optimizer.step()
            
        # lr_scheduler.step()
        val_loss, _, _ = evaluate(model, 'val')
        end_time2 = time.time()
        message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch_num, 
                   args.epochs, batches_seen, np.mean(losses), val_loss, optimizer.param_groups[0]['lr'], (end_time2 - start_time))
        logger.info(message)
            
        # if (epoch_num % args.test_every_n_epochs) == args.test_every_n_epochs - 1:
        #     test_loss, ys_true, ys_pred = evaluate(model, 'test')
        #     evaluate_metric(ys_true, ys_pred)

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            # logger.info('Val loss decrease from {:.4f} to {:.4f}, saving model to pt'.format(min_val_loss, val_loss))
            torch.save(model.state_dict(), modelpt_path)
        elif val_loss >= min_val_loss:
            wait += 1
            if wait == args.patience:
                logger.info('Early stopping at epoch: %d' % epoch_num)
                break
    
    logger.info('=' * 35 + 'Best model performance' + '=' * 35)
    model = get_model()
    model = init_model(model)
    model.load_state_dict(torch.load(modelpt_path))
    test_loss, ys_true, ys_pred = evaluate(model, 'test')
    evaluate_metric(ys_true, ys_pred)

#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY'], default='METRLA', help='which dataset to run')
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the ratio of training and validation data among the total')
parser.add_argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--num_nodes', type=int, default=207, help='num_nodes')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--horizon', type=int, default=12, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=2, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
parser.add_argument('--num_rnn_layers', type=int, default=1, help='number of rnn layers')
parser.add_argument('--rnn_units', type=int, default=64, help='number of rnn units')
parser.add_argument('--filter_type', type=str, default='dual_random_walk', help='filter_type')
parser.add_argument('--max_diffusion_step', type=int, default=3, help='max_diffusion_step')
parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--patience", type=int, default=100, help="patience used for early stop")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--base_lr", type=float, default=0.005, help="base learning rate")
parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
parser.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning")
parser.add_argument("--steps", type=eval, default=[20, 30, 40], help="steps")
parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
parser.add_argument('--knn_k', type=int, default=10, help='knn_k')
parser.add_argument('--dim_fc', type=int, default=383552, help='dim_fc')
parser.add_argument('--test_every_n_epochs', type=int, default=5, help='test_every_n_epochs')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
# parser.add_argument('--seed', type=int, default=100, help='random seed.')
args = parser.parse_args()
        
if args.dataset == 'METRLA':
    data_path = f'../{args.dataset}/metr-la.h5'
    args.num_nodes = 207
    args.rnn_units = 64
    args.max_diffusion_step = 3
    args.base_lr = 0.005
    args.knn_k = 10
    args.dim_fc = 383552
elif args.dataset == 'PEMSBAY':
    data_path = f'../{args.dataset}/pems-bay.h5'
    args.num_nodes = 325
    args.rnn_units = 128
    args.max_diffusion_step = 2
    args.base_lr = 0.001
    args.knn_k = 30
    args.dim_fc = 583408
else:
    pass # including more datasets in the future    

model_name = 'GTS'
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'../save/{args.dataset}_{model_name}_{timestring}'
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
score_path = f'{path}/{model_name}_{timestring}_scores.txt'
epochlog_path = f'{path}/{model_name}_{timestring}_epochlog.txt'
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
if not os.path.exists(path): os.makedirs(path)
shutil.copy2(sys.argv[0], path)
shutil.copy2('GTS.py', path)
shutil.copy2('utils.py', path)
    
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
logger.info('trainval_ratio', args.trainval_ratio)
logger.info('val_ratio', args.val_ratio)
logger.info('num_nodes', args.num_nodes)
logger.info('seq_len', args.seq_len)
logger.info('horizon', args.horizon)
logger.info('input_dim', args.input_dim)
logger.info('output_dim', args.output_dim)
logger.info('num_rnn_layers', args.num_rnn_layers)
logger.info('rnn_units', args.rnn_units)
logger.info('loss', args.loss)
logger.info('epochs', args.epochs)
logger.info('batch_size', args.batch_size)
logger.info('base_lr', args.base_lr)
logger.info('use_curriculum_learning', args.use_curriculum_learning)
logger.info('knn_k', args.knn_k)
logger.info('dim_fc', args.dim_fc)

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

data = {}
for category in ['train', 'val', 'test']:
    cat_data = np.load(os.path.join(f'../{args.dataset}', category + '.npz'))
    data['x_' + category] = cat_data['x']
    data['y_' + category] = cat_data['y']
scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
# Data format, data['x_train'], data['y_train'], data['x_val'], data['y_val'], data['x_test'], data['y_test']
for category in ['train', 'val', 'test']:
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
data['train_loader'] = DataLoader(data['x_train'], data['y_train'], args.batch_size, shuffle=True)
data['val_loader'] = DataLoader(data['x_val'], data['y_val'], args.batch_size, shuffle=False)
data['test_loader'] = DataLoader(data['x_test'], data['y_test'], args.batch_size, shuffle=False)
    
df = pd.read_hdf(data_path).values
train_feas = df[:int(df.shape[0]*args.trainval_ratio*(1 - args.val_ratio))] # 0.8*(1-0.125)=0.7
scaler1 = StandardScaler(mean=train_feas.mean(), std=train_feas.std())
train_feas = scaler1.transform(train_feas)
print('scaler.mean, scaler.std, scaler1.mean, scaler1.std', scaler.mean, scaler.std, scaler1.mean, scaler1.std)

from sklearn.neighbors import kneighbors_graph
g = kneighbors_graph(train_feas.T, args.knn_k, metric='cosine')
g = np.array(g.todense(), dtype=np.float32)
adj_mx = torch.Tensor(g).to(device)
train_feas = torch.Tensor(train_feas).to(device)

def main():
    logger.info(args.dataset, 'training and testing started', time.ctime())
    logger.info('adj_mx.shape, train_feas.shape', adj_mx.shape, train_feas.shape)
    logger.info('train xs.shape, ys.shape', data['x_train'].shape, data['y_train'].shape)
    logger.info('val xs.shape, ys.shape', data['x_val'].shape, data['y_val'].shape)
    logger.info('test xs.shape, ys.shape', data['x_test'].shape, data['y_test'].shape)
    traintest_model()
    logger.info(args.dataset, 'training and testing ended', time.ctime())
    
if __name__ == '__main__':
    main()


# base_dir: data/model
# log_level: INFO

# data:
#   batch_size: 64
#   dataset_dir: data/METR-LA
#   test_batch_size: 64
#   val_batch_size: 64
#   graph_pkl_filename: data/sensor_graph/adj_mx.pkl

# model:
#   cl_decay_steps: 2000
#   filter_type: dual_random_walk
#   horizon: 12
#   input_dim: 2
#   l1_decay: 0 # not used
#   max_diffusion_step: 3
#   num_nodes: 207
#   num_rnn_layers: 1
#   output_dim: 1
#   rnn_units: 64
#   seq_len: 12
#   use_curriculum_learning: true
#   dim_fc: 383552

# train:
#   base_lr: 0.005 # done
#   optimizer: adam # done
#   epochs: 200 # done
#   epsilon: 1.0e-3 # done
#   lr_decay_ratio: 0.1 # done
#   max_grad_norm: 5 # done
#   patience: 100 # done
#   steps: [20, 30, 40] # done
#   test_every_n_epochs: 5 # done
#   knn_k: 10 done
#   epoch_use_regularization: 200 # not used
#   dropout: 0 # not used
#   epoch: 0 # not used
#   global_step: 0 # not used
#   max_to_keep: 100 # not used
#   min_learning_rate: 2.0e-06 not sued
#   num_sample: 10 # not used