# ICWSM'25
Influence Maximization in Temporal Social Networks with a Cold-start Problem:
A Supervised Approach

[[Paper]()]

## Preparation

1. Install dependencies:
```bash
pip install -r requirements.txt
```
DGL installation:
```bash
pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html
```

2. Data downloading and run preprocessing
```bash
cd data_processing && python3 read_tweet.py
```

Directory tree:
```bash
.
├── checkpoint
├── data_processing
│   ├── DGLgraph
│   ├── netease_week
│   └── sampled_week
├── inference
├── labeling
└── models
    ├── cold_start
    ├── graphmae
    └── tgn_raw
```

2. Labeling
```bash
python3 labeling/label.py
```
see README in labeling

3. Running Cold-start algorithm
```bash
python3 models/cold_start/cold_start.py
```

## Offline Training
Task: node classification 

Metrics: accuracy 

Data: \
./data_processing/netease_week/*_graphs.csv, 
./data_processing/netease_week/*_features.csv, 
./data_processing/netease_week/*_labels.csv

Output: \
./checkpoint/netease_graphmae_best_model.ckpt
```bash
python run_all_model.py --task train_offline --supervised
```

## Online Training
Same with the offline training, except that no test data is splited
```bash
python run_all_model.py --task train_online --supervised
```

## Online Inference
Task: Influence Maximization \
Given K users at first that will be recommended in priority, how many users will undergo invitation/adoption at least once after T times.

Data: \
./data_processing/netease_week/*_graphs.csv, 
./data_processing/netease_week/*_features.csv

Model checkpoint: \
./checkpoint/netease_graphmae_best_model.ckpt

Metrics: \
daily number of users that have invitation or adoption

Output: \
./inference/netease_graphmae_emb.npy
./inference/netease_graphmae_pred.npy
```bash
python run_all_model.py --task inference_online --supervised --inference_data netease_week
```

## train TGN
```bash
python models/tgn_raw/train_model.py --use_memory --prefix tgn-attn --n_runs 1 --n_epoch 10 --bs 64 --message_dim 50 --memory_dim 86 --platform netease
```
