import sys
import os
import shutil
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import argparse
import logging
from utils import StandardScaler, DataLoader, masked_mae_loss, masked_mape_loss, masked_mse_loss, masked_rmse_loss
from MegaCRNA import MegaCRN

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
    model = MegaCRN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, horizon=args.horizon, 
                    rnn_units=args.rnn_units, num_layers=args.num_rnn_layers, mem_num=args.mem_num, mem_dim=args.mem_dim, cheb_k = args.max_diffusion_step,
                    memory_type=args.memory, meta_type=args.meta, cl_decay_steps=args.cl_decay_steps, use_curriculum_learning=args.use_curriculum_learning).to(device)
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
    x0 = x[..., :args.input_dim]
    y0 = y[..., :args.output_dim]
    y1 = y[..., args.output_dim:]
    x0 = torch.from_numpy(x0).float()
    y0 = torch.from_numpy(y0).float()
    y1 = torch.from_numpy(y1).float()
    return x0.to(device), y0.to(device), y1.to(device) # x, y, y_cov
    
def evaluate(model, mode):
    with torch.no_grad():
        model = model.eval()
        data_iter =  data[f'{mode}_loader'].get_iterator()
        losses = []
        ys_true, ys_pred = [], []
        maes, mapes, mses = [], [], []
        l_3, m_3, r_3 = [], [], []
        l_6, m_6, r_6 = [], [], []
        l_12, m_12, r_12 = [], [], []
        for x, y in data_iter:
            x, y, ycov = prepare_x_y(x, y)
            output, h_att, query, pos, neg = model(x, ycov)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)
            loss1 = masked_mae_loss(y_pred, y_true) # masked_mae_loss(y_pred, y_true)
            separate_loss = nn.TripletMarginLoss(margin=1.0)
            compact_loss = nn.MSELoss()
            loss2 = separate_loss(query, pos, neg)
            loss3 = compact_loss(query, pos)
            loss = loss1 + args.lamb * loss2 + args.lamb1 * loss3
            losses.append(loss.item())
            # Followed the DCRNN TensorFlow Implementation
            maes.append(masked_mae_loss(y_pred, y_true).item())
            mapes.append(masked_mape_loss(y_pred, y_true).item())
            mses.append(masked_mse_loss(y_pred, y_true).item())
            # Important for MegaCRN model to let T come first.
            y_true, y_pred = y_true.permute(1, 0, 2, 3), y_pred.permute(1, 0, 2, 3)
            l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
            m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
            r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
            l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
            m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
            r_6.append(masked_mse_loss(y_pred[5:6], y_true[5:6]).item())
            l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
            m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
            r_12.append(masked_mse_loss(y_pred[11:12], y_true[11:12]).item())
            ys_true.append(y_true)
            ys_pred.append(y_pred)
        mean_loss = np.mean(losses)
        mean_mae, mean_mape, mean_rmse = np.mean(maes), np.mean(mapes), np.sqrt(np.mean(mses))
        l_3, m_3, r_3 = np.mean(l_3), np.mean(m_3), np.sqrt(np.mean(r_3))
        l_6, m_6, r_6 = np.mean(l_6), np.mean(m_6), np.sqrt(np.mean(r_6))
        l_12, m_12, r_12 = np.mean(l_12), np.mean(m_12), np.sqrt(np.mean(r_12))
        if mode == 'test':
            logger.info('Horizon overall: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_mae, mean_mape, mean_rmse))
            logger.info('Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_3, m_3, r_3))
            logger.info('Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_6, m_6, r_6))
            logger.info('Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_12, m_12, r_12))
        return mean_loss, ys_true, ys_pred
        
def traintest_model():  
    model = get_model()
    print_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, eps=args.epsilon)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    min_val_loss = float('inf')
    wait = 0
    batches_seen = 0
    for epoch_num in range(args.epochs):
        start_time = time.time()
        model = model.train()
        data_iter = data['train_loader'].get_iterator()
        losses = []
        for x, y in data_iter:
            optimizer.zero_grad()
            x, y, ycov = prepare_x_y(x, y)
            output, h_att, query, pos, neg = model(x, ycov, y, batches_seen)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)
            loss1 = masked_mae_loss(y_pred, y_true) # masked_mae_loss(y_pred, y_true)
            separate_loss = nn.TripletMarginLoss(margin=1.0)
            compact_loss = nn.MSELoss()
            loss2 = separate_loss(query, pos, neg)
            loss3 = compact_loss(query, pos)
            loss = loss1 + args.lamb * loss2 + args.lamb1 * loss3
            losses.append(loss.item())
            batches_seen += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # gradient clipping - this does it in place
            optimizer.step()
        train_loss = np.mean(losses)
        lr_scheduler.step()
        val_loss, _, _ = evaluate(model, 'val')
        # if (epoch_num % args.test_every_n_epochs) == args.test_every_n_epochs - 1:
        end_time2 = time.time()
        message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch_num + 1, 
                   args.epochs, batches_seen, train_loss, val_loss, optimizer.param_groups[0]['lr'], (end_time2 - start_time))
        logger.info(message)
        test_loss, _, _ = evaluate(model, 'test')
        
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
            # logger.info('Val loss decrease from {:.4f} to {:.4f}, saving model to pt'.format(min_val_loss, val_loss))
        elif val_loss >= min_val_loss:
            wait += 1
            if wait == args.patience:
                logger.info('Early stopping at epoch: %d' % epoch_num)
                break
    
    logger.info('=' * 35 + 'Best model performance' + '=' * 35)
    model = get_model()
    model.load_state_dict(torch.load(modelpt_path))
    test_loss, _, _ = evaluate(model, 'test')

#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY'], default='METRLA', help='which dataset to run')
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the ratio of training and validation data among the total')
parser.add_argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--num_nodes', type=int, default=207, help='num_nodes')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--horizon', type=int, default=12, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
parser.add_argument('--max_diffusion_step', type=int, default=3, help='max diffusion step or Cheb K')
parser.add_argument('--num_rnn_layers', type=int, default=1, help='number of rnn layers')
parser.add_argument('--rnn_units', type=int, default=64, help='number of rnn units')
parser.add_argument('--mem_num', type=int, default=207, help='number of meta-nodes/prototypes')
parser.add_argument('--mem_dim', type=int, default=64, help='dimension of meta-nodes/prototypes')
parser.add_argument("--memory", type=eval, choices=[True, False], default='True', help="whether to use memory: True or False")
parser.add_argument("--meta", type=eval, choices=[True, False], default='True', help="whether to use meta-graph: True or False")
parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
parser.add_argument('--lamb', type=float, default=0.01, help='lamb value for separate loss')
parser.add_argument('--lamb1', type=float, default=0.01, help='lamb1 value for compact loss')
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--patience", type=int, default=20, help="patience used for early stop")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--base_lr", type=float, default=0.01, help="base learning rate")
parser.add_argument("--steps", type=eval, default=[50, 100, 150], help="steps")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
parser.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning")
parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
parser.add_argument('--test_every_n_epochs', type=int, default=5, help='test_every_n_epochs')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
# parser.add_argument('--seed', type=int, default=100, help='random seed.')
args = parser.parse_args()
        
if args.dataset == 'METRLA':
    data_path = f'../{args.dataset}/metr-la.h5'
    args.num_nodes = 207
    args.mem_num = args.num_nodes
    # args.rnn_units = 64
    # args.max_diffusion_step = 3
    # args.base_lr = 0.005
elif args.dataset == 'PEMSBAY':
    data_path = f'../{args.dataset}/pems-bay.h5'
    args.num_nodes = 325
    args.mem_num = args.num_nodes
    args.max_diffusion_step = 2
    # args.rnn_units = 128
    # args.base_lr = 0.001
else:
    pass # including more datasets in the future    

model_name = 'MegaCRNA'
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'../save/{args.dataset}_{model_name}_{timestring}'
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
score_path = f'{path}/{model_name}_{timestring}_scores.txt'
epochlog_path = f'{path}/{model_name}_{timestring}_epochlog.txt'
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
if not os.path.exists(path): os.makedirs(path)
shutil.copy2(sys.argv[0], path)
shutil.copy2('MegaCRNA.py', path)
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
logger.info('max_diffusion_step', args.max_diffusion_step)
logger.info('mem_num', args.mem_num)
logger.info('mem_dim', args.mem_dim)
logger.info('memory', args.memory)
logger.info('meta', args.meta)
logger.info('loss', args.loss)
logger.info('separate loss lamb', args.lamb)
logger.info('compact loss lamb1', args.lamb1)
logger.info('epochs', args.epochs)
logger.info('batch_size', args.batch_size)
logger.info('base_lr', args.base_lr)
logger.info('steps', args.steps)
logger.info('lr_decay_ratio', args.lr_decay_ratio)
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

data = {}
for category in ['train', 'val', 'test']:
    cat_data = np.load(os.path.join(f'../{args.dataset}', category + '.npz'))
    data['x_' + category] = cat_data['x']
    data['y_' + category] = cat_data['y']
scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
for category in ['train', 'val', 'test']:
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
data['train_loader'] = DataLoader(data['x_train'], data['y_train'], args.batch_size, shuffle=True)
data['val_loader'] = DataLoader(data['x_val'], data['y_val'], args.batch_size, shuffle=False)
data['test_loader'] = DataLoader(data['x_test'], data['y_test'], args.batch_size, shuffle=False)

def main():
    logger.info(args.dataset, 'training and testing started', time.ctime())
    logger.info('train xs.shape, ys.shape', data['x_train'].shape, data['y_train'].shape)
    logger.info('val xs.shape, ys.shape', data['x_val'].shape, data['y_val'].shape)
    logger.info('test xs.shape, ys.shape', data['x_test'].shape, data['y_test'].shape)
    traintest_model()
    logger.info(args.dataset, 'training and testing ended', time.ctime())
    
if __name__ == '__main__':
    main()