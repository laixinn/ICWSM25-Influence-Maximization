import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
import argparse, datetime

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import random
import pickle
import logging, time
from tqdm import tqdm
from sklearn.metrics import f1_score

from data_processing.read_tweet import build_tgraph, read_nodes
from models.dataloader import DGLData, DGLDataInf

## graphmae
from models.graphmae.models import build_graphmae
from models.graphmae.evaluation import evaluete, inference, supervised_evaluete
## tgn
from models.tgn_raw.train_model import load_and_process

from pathlib import Path
import logging

# debugging cuda
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

def load_offline_netease(edges, platform='local', batch_size=128, is_cold=False):
    node_features, src_labels, dst_labels, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
        new_node_test_data, n_node, node_to_index, num_classes = load_and_process(True, reindex=True, elist=edges, platform=platform, is_cold=is_cold)
    num_features = node_features.shape[1]
    
    # build Dataset
    train_dataset = DGLData(node_features, train_data, batch_size=batch_size)
    val_dataset = DGLData(node_features, val_data, batch_size=batch_size)
    test_dataset = DGLData(node_features, test_data, batch_size=batch_size)

    # build dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=train_dataset.collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=val_dataset.collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn, shuffle=True)

    return train_loader, val_loader, test_loader, num_features, num_classes

def load_online_netease_v2(edges, platform='local', batch_size=128, is_cold=False):
    node_features, src_labels, dst_labels, edge_features, full_data, train_data, _, val_data, _, \
        new_node_val_data, n_node, node_to_index, num_classes = load_and_process(False, reindex=True, elist=edges, platform=platform, is_cold=is_cold)
    num_features = node_features.shape[1]
    
    # build Dataset
    train_dataset = DGLData(node_features, train_data, batch_size=batch_size)
    val_dataset = DGLData(node_features, val_data, batch_size=batch_size)

    # build dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=train_dataset.collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=val_dataset.collate_fn, shuffle=True)

    return train_loader, val_loader, None, num_features, num_classes

def run_static_twitter_online(args):
    path = args.data_path
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    epoch = args.epoch
    device = args.cuda
    supervised = True# args.supervised
    EDGES = list(map(int, args.edges.split(',')))
    
    # random seed
    np.random.seed(1)
    random.seed(1)

    # load Dataset
    # train_loader, num_features, num_classes = \
    #     load_online_netease(False, edges=EDGES, batch_size=batch_size, is_cold=args.cold)
    train_loader, val_loader, _, num_features, num_classes = \
        load_online_netease_v2(edges=EDGES, batch_size=batch_size, is_cold=args.cold)

    # model init
    model = build_graphmae('models/graphmae/configs.yml', num_features, supervised=supervised, n_classes=num_classes)

    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    # start train
    lowest_auc = 1e5
    save_str = f'checkpoint/netease_graphmae_{args.cold}'
    for epo in range(epoch):
        # train
        epoch_list = []
        model.train()
        for subgraph in train_loader:
            optimizer.zero_grad()

            subgraph = subgraph[0].to(device)
            if not supervised:
                loss, loss_dict = model(subgraph, subgraph.ndata['feat'])
            else:
                loss, loss_dict = model(subgraph, subgraph.ndata['feat'], subgraph.ndata['label'])

            loss.backward()
            optimizer.step()

            epoch_list.append(loss.clone().detach().cpu().numpy())
        epoch_loss = np.mean(epoch_list)
        print('Epoch %d: %.4f avg loss' % (epo, epoch_loss))

        if (epo >= epoch // 2) and (epo % 10 == 0):
            val_1, val_acc, val_ap, val_auc, _, _, _, _ = supervised_evaluete(model, (train_loader, val_loader, val_loader),
                 num_classes, save_str='checkpoint/tmp_netease_graphmae', device=device)
            if val_auc < lowest_auc:
                lowest_auc = val_auc
                torch.save({
                    'encoder_state_dict': model.state_dict(),
                    'gnn_state_dict': model.state_dict()
                }, '%s_best_model.ckpt' % (save_str))
                print(f'Save model at epoch {epo} with loss {val_auc}')
                
    torch.save({
        'encoder_state_dict': model.state_dict(),
        'gnn_state_dict': model.state_dict()
    }, '%s_best_model.ckpt' % (save_str))
    print(f'Save model at epoch {epo} with loss {lowest_auc}')

    # whether to update model
    # if os.path.exists('checkpoint/netease_graphmae_best_model.ckpt'):
    #     try:
    #         ckpt = torch.load('checkpoint/netease_graphmae_best_model.ckpt')
    #         model.load_state_dict(ckpt['gnn_state_dict'])
    #         if not supervised:
    #             evaluete(model, (train_loader, val_loader, test_loader),
    #                  num_classes, save_str='checkpoint/netease_graphmae', device=device)
    #         else:
    #             supervised_evaluete(model, (train_loader, val_loader, test_loader),
    #                  num_classes, save_str='checkpoint/netease_graphmae', device=device)
    #     except:
    #         print('ckpt not compatible, update model.')
    #         os.system('cp %s %s' % ('checkpoint/tmp_netease_graphmae_best_model.ckpt', 'checkpoint/netease_graphmae_best_model.ckpt'))
    # else:
    #     os.system('cp %s %s' % ('checkpoint/tmp_netease_graphmae_best_model.ckpt', 'checkpoint/netease_graphmae_best_model.ckpt'))
    
def graphmae_netease_offline(args):
    path = args.data_path
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    epoch = args.epoch
    device = args.cuda
    supervised = True# args.supervised
    EDGES = list(map(int, args.edges.split(',')))
    
    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    filepath = 'models/graphmae'
    Path(f"{filepath}/log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('{}/log/{}.log'.format(filepath, str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)
    
    # random seed
    np.random.seed(1)
    random.seed(1)

    # load Dataset
    train_loader, val_loader, test_loader, num_features, num_classes = \
        load_offline_netease(edges=EDGES, batch_size=batch_size, is_cold=args.cold)

    # model init
    model = build_graphmae('models/graphmae/configs.yml', num_features, supervised=supervised, n_classes=num_classes)

    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    # start train
    lowest_loss = 1e5
    save_str = 'checkpoint/netease_graphmae_best_model.ckpt'
    for epo in tqdm(range(epoch)):
        # train
        epoch_list = []
        model.train()
        for subgraph in train_loader:
            optimizer.zero_grad()

            subgraph = subgraph[0].to(device)
            loss, loss_dict = model(subgraph, subgraph.ndata['feat'], subgraph.ndata['label'])

            loss.backward()
            optimizer.step()

            epoch_list.append(loss.clone().detach().cpu().numpy())
        epoch_loss = np.mean(epoch_list)
        logger.info('Epoch %d: %.4f avg loss' % (epo, epoch_loss))

        if epo % 10 == 0:
            # if not supervised:
            #     evaluete(model, (train_loader, val_loader, test_loader),
            #          num_classes, save_str='checkpoint/tmp_netease_graphmae', device=device)
            # else:
            val_1, val_acc, val_ap, val_auc, test_1, test_acc, test_ap, test_auc = supervised_evaluete(model, (train_loader, val_loader, test_loader),
                 num_classes, save_str=f'checkpoint/tmp_netease_graphmae_{args.cold}', device=device)
            logger.info(f"# val_1: {val_1: .4f}, val_acc:{val_acc: .4f}, val_ap:{val_ap: .4f}, val_auc:{val_auc: .4f}, "
                f"test_1: {test_1: .4f}, test_acc:{test_acc: .4f}, test_ap:{test_ap: .4f}, test_auc:{test_auc: .4f}")
            
def graphmae_netease_evaluation(args):
    path = args.data_path
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    epoch = args.epoch
    device = args.cuda
    supervised = True# args.supervised
    EDGES = list(map(int, args.edges.split(',')))
    
    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    filepath = 'models/graphmae'
    Path(f"{filepath}/log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('{}/log/{}.log'.format(filepath, str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)
    
    # random seed
    np.random.seed(1)
    random.seed(1)

    # load Dataset
    train_loader, val_loader, test_loader, num_features, num_classes = \
        load_offline_netease(edges=EDGES, batch_size=batch_size, is_cold=args.cold)

    # model init
    model = build_graphmae('models/graphmae/configs.yml', num_features, supervised=supervised, n_classes=num_classes)

    model = model.to(device)
    # load model
    ckpt = torch.load('%s_best_model.ckpt' % (f'checkpoint/tmp_netease_graphmae_{args.cold}'))
    model.load_state_dict(ckpt['gnn_state_dict'])

    val_1, val_acc, val_ap, val_auc, test_1, test_acc, test_ap, test_auc = supervised_evaluete(model, (train_loader, val_loader, test_loader),
         num_classes, save_str=f'checkpoint/tmp_netease_graphmae_{args.cold}', device=device)
    logger.info(f"# val_1: {val_1: .4f}, val_acc:{val_acc: .4f}, val_ap:{val_ap: .4f}, val_auc:{val_auc: .4f}, "
        f"test_1: {test_1: .4f}, test_acc:{test_acc: .4f}, test_ap:{test_ap: .4f}, test_auc:{test_auc: .4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--task', type=str, default='train_offline')
    parser.add_argument('--cuda', type=str, default='cuda')
    # parser.add_argument('--supervised', type=int, default=1)
    parser.add_argument('--inference_data', type=str, default='tweet_week')
    parser.add_argument('--save_noah', action='store_true')
    parser.add_argument('--data_path', type=str, default='tweet_week')
    parser.add_argument('--model_name', type=str, default='tgn', choices=['graphmae', 'tgn', 'g_e'], help='support model: graphmae, tgn, g_e')
    parser.add_argument('--platform', type=str, default='local')
    parser.add_argument('--edges', type=str, default='0,1', help='Which graph to use')
    parser.add_argument('--cold', action='store_true', help='Whether to add cold-start edges')

    args, _ = parser.parse_known_args()
    
    if args.cold and not '2' in args.edges:
        args.edges += ',2'
    print(args)
    
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    if not os.path.exists('inference'):
        os.makedirs('inference')
        
    if args.task == 'train_online':
        if args.model_name == 'graphmae':
            '''tweet graphmae'''
            run_static_twitter_online(args)
        elif args.model_name == 'tgn':
            pass
        elif args.model_name == 'g_e':
            pass
        else:
            raise NotImplementedError
    elif args.task == 'inference_online':
        raise NotImplementedError
    elif args.task == 'train_offline':
        if args.model_name == 'graphmae':
            '''tweet graphmae'''
            graphmae_netease_offline(args)
        elif args.model_name == 'tgn':
            raise NotImplementedError
    elif args.task == 'evaluate_offline':
        if args.model_name == 'graphmae':
            '''tweet graphmae'''
            graphmae_netease_evaluation(args)

