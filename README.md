### MegaCRN: Meta-Graph Convolutional Recurrent Network

#### [AAAI23] R. Jiang*, Z. Wang*, J. Yong, P. Jeph, Q. Chen, Y. Kobayashi, X. Song, S. Fukushima, T. Suzumura, "Spatio-Temporal Meta-Graph Learning for Traffic Forecasting", Proc. of Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI), 2023. (Accepted to Appear)
#### [AI23] R. Jiang*, Z. Wang*, J. Yong, P. Jeph, Q. Chen, Y. Kobayashi, X. Song, T. Suzumura, S. Fukushima, "MegaCRN: Meta-Graph Convolutional Recurrent Network for Spatio-Temporal Modeling", Artificial intelligence, 2023. (Extended Journal Version under Review)

#### Code and data are now available (more data will come). Please kindly cite the following bibtex. 
@article{jiang2022spatio,
  title={Spatio-Temporal Meta-Graph Learning for Traffic Forecasting},
  author={Jiang, Renhe and Wang, Zhaonan and Yong, Jiawei and Jeph, Puneet and Chen, Quanjun and Kobayashi, Yasumasa and Song, Xuan and Fukushima, Shintaro and Suzumura, Toyotaro},
  journal={arXiv preprint arXiv:2211.14701},
  year={2022}
}

#### Preprints

[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=MegaCRN&color=red&logo=arxiv)](https://arxiv.org/abs/2211.14701)
[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=MegaCRN(journal)&color=red&logo=arxiv)](https://arxiv.org/abs/2212.05989)

#### Performance on Traffic Speed Benchmarks

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-meta-graph-learning-for/traffic-prediction-on-metr-la)](https://paperswithcode.com/sota/traffic-prediction-on-metr-la?p=spatio-temporal-meta-graph-learning-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-meta-graph-learning-for/traffic-prediction-on-pems-bay)](https://paperswithcode.com/sota/traffic-prediction-on-pems-bay?p=spatio-temporal-meta-graph-learning-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-meta-graph-learning-for/traffic-prediction-on-expy-tky)](https://paperswithcode.com/sota/traffic-prediction-on-expy-tky?p=spatio-temporal-meta-graph-learning-for)

#### Performance on Traffic Flow Benmarks

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/megacrn-meta-graph-convolutional-recurrent/traffic-prediction-on-pemsd3)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd3?p=megacrn-meta-graph-convolutional-recurrent)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/megacrn-meta-graph-convolutional-recurrent/traffic-prediction-on-pemsd4)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd4?p=megacrn-meta-graph-convolutional-recurrent)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/megacrn-meta-graph-convolutional-recurrent/traffic-prediction-on-pemsd7)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd7?p=megacrn-meta-graph-convolutional-recurrent)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/megacrn-meta-graph-convolutional-recurrent/traffic-prediction-on-pemsd8)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd8?p=megacrn-meta-graph-convolutional-recurrent)

#### Requirements
* Python 3.8.8 -> Anaconda Distribution
* pytorch 1.9.1 -> py3.8_cuda11.1_cudnn8.0.5_0
* pandas 1.2.4 
* numpy 1.20.1
* torch-summary 1.4.5 -> pip install torch-summary https://pypi.org/project/torch-summary/ (must necessary)
* jpholiday -> pip install jpholiday (not must, but if you want onehottime)

#### General Description
* The directory is structured in a flat style and only with two levels.
* The datasets are stored in DATA directories, and the model codes are put in model_DATA directories. 
* The training and testing function is merged into one file, we can just run "python traintest_MegaCRN.py" under each model directory.
* Also we can run "python MegaCRN.py" to simply check the model architecture without feeding the data.
* Also under model directory, metrics.py file contains the metric functions and utils.py file contains a set of supporting functions.

##### How to run our model (general command)?
* cd model
* python traintest_MegaCRN.py --dataset=DATA --gpu=GPU_DEVICE_ID 
* DATA = {METRLA, PEMSBAY}
* For PEMSBAY dataset, please first upzip ./PEMSBAY/pems-bay.zip to get ./PEMSBAY/pems-bay.h5 file.

##### How to run our model on PEMS03,04,07,08?
* cd model_PEMS3478
* python traintest_MegaCRN.py --dataset=DATA --gpu=GPU_DEVICE_ID 
* DATA = {PEMS03, PEMS04, PEMS07, PEMS08}

##### How to run our model on our data EXPY-TKY?
* cd model_EXPYTKY
* python traintest_MegaCRN.py --dataset=DATA --gpu=GPU_DEVICE_ID 
* DATA = {EXPYTKY, EXPYTKY*} 
* EXPYTKY with 1843 links is the data used in our paper; EXPYTKY* is a superset of EXPYTKY that contains all 2841 expy-tky links.

#### Arguments (Default)
The default hyperparameters used in our paper are written in model/traintest_MegaCRN.py as follows.
* argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY'], default='METRLA', help='which dataset to run')
* argument('--trainval_ratio', type=float, default=0.8, help='the ratio of training and validation data among the total')
* argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
* argument('--seq_len', type=int, default=12, help='sequence length of prediction')
* argument('--his_len', type=int, default=12, help='sequence length of historical observation')
* argument('--channelin', type=int, default=1, help='number of input channel')
* argument('--channelout', type=int, default=1, help='number of output channel')
* argument("--loss", type=str, default='MaskMAE', help="MAE, MSE, MaskMAE")
* argument("--epoch", type=int, default=200, help="number of epochs of training")
* argument("--batch_size", type=int, default=64, help="size of the batches")
* argument("--lr", type=float, default=0.001, help="adam: learning rate")
* argument("--patience", type=float, default=10, help="patience used for early stop")
* argument('--num_layers', type=int, default=1, help='number of layers')
* argument('--hiddenunits', type=int, default=32, help='number of hidden units')
* argument('--mem_num', type=int, default=10, help='number of meta-nodes/prototypes')
* argument('--mem_dim', type=int, default=32, help='dimension of meta-nodes/prototypes')
* argument("--memory", type=eval, choices=[True, False], default='True', help="whether to use memory: True or False")
* argument("--meta", type=eval, choices=[True, False], default='True', help="whether to use meta-graph: True or False")
* argument("--decoder", type=str, choices=['sequence', 'stepwise'], default='stepwise', help="decoder type: sequence or stepwise")
* argument('--lamb', type=float, default=0.01, help='lamb value for separate loss')
* argument('--lamb1', type=float, default=0.01, help='lamb1 value for compact loss')
* argument('--gpu', type=int, default=0, help='which gpu to use')
* argument('--seed', type=int, default=100, help='random seed.')

#### Arguments (PEMS03,04,07,08)
The hyperparameters for PEMS03,04,07,08 in model_PEMS3478/traintest_MegaCRN.py are the same as the above except: 
* argument("--loss", type=str, default='MAE', help="MAE, MSE, MaskMAE")
* argument("--patience", type=float, default=200, help="patience used for early stop")
