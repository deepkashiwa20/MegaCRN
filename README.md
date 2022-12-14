### MegaCRN

#### [AAAI23] R. Jiang*, Z. Wang*, J. Yong, P. Jeph, Q. Chen, Y. Kobayashi, X. Song, S. Fukushima, T. Suzumura, "Spatio-Temporal Meta-Graph Learning for Traffic Forecasting", Proc. of Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI), 2023. (Accepted to Appear)
#### [AI23] R. Jiang*, Z. Wang*, J. Yong, P. Jeph, Q. Chen, Y. Kobayashi, X. Song, T. Suzumura, S. Fukushima, "MegaCRN: Meta-Graph Convolutional Recurrent Network for Spatio-Temporal Modeling", Artificial intelligence, 2023. (Extended Journal Version under Review)

#### Code and data are now available. If you find our work useful, please kindly cite the following. 
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

## General Description
* The directory is structured in a flat style and only with two levels.
* The datasets are stored in DATA directories, and the model codes are put in model_DATA directories. 
* The training and testing function is merged into one file, we can just run "python traintest_MegaCRN.py" under each model directory.
* Also we can run "python MegaCRN.py" to simply check the model architecture without feeding the training data under each model directory.
* Also in the model directories, metrics.py file contains the metric functions and utils.py file contains a set of supporting functions.


## Requirements
* Python 3.8.8 -> Anaconda Distribution
* pytorch 1.9.1 -> py3.8_cuda11.1_cudnn8.0.5_0
* pandas 1.2.4 
* numpy 1.20.1
* torch-summary 1.4.5 -> pip install torch-summary https://pypi.org/project/torch-summary/ (must necessary)
* jpholiday -> pip install jpholiday (not must, but if you want onehottime)

## How to run our model (general)?

* cd model
* python traintest_MegaCRN.py --dataset=DATA --gpu=YOUR_GPU_DEVICE_ID 
* DATA = {METRLA, PEMSBAY}
* For PEMSBAY dataset, please first upzip ./PEMSBAY/pems-bay.zip to get ./PEMSBAY/pems-bay.h5 file.

## How to run our model on our data EXPY-TKY?


## Arguments
* The hyperparameters used in our paper have been written in each ./model_DATA/traintest_MegaCRN.py file as follows.
* parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
* parser.add_argument("--epoch", type=int, default=200, help="number of epochs of training")
* parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
* parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
* parser.add_argument("--patience", type=float, default=10, help="patience used for early stop")
* parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the ratio of training and validation data among the total')
* parser.add_argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
* parser.add_argument('--seq_len', type=int, default=12, help='sequence length of prediction')
* parser.add_argument('--his_len', type=int, default=12, help='sequence length of historical observation')
* parser.add_argument('--city', type=str, default='METRLA', help='which dataset')
* parser.add_argument('--channelin', type=int, default=1, help='number of input channel')
* parser.add_argument('--channelout', type=int, default=1, help='number of output channel')
* parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
* parser.add_argument('--hiddenunits', type=int, default=32, help='number of hidden units')
* parser.add_argument('--mem_num', type=int, default=10, help='number of memory')
* parser.add_argument('--mem_dim', type=int, default=32, help='dimension of memory')
* parser.add_argument('--lamb', type=float, default=0.01, help='lamb value for separate loss')
* parser.add_argument('--lamb1', type=float, default=0.01, help='lamb1 value for compact loss')
