import math
import logging
import time
import sys
import argparse
from tqdm import tqdm

import pandas as pd
import torch
import numpy as np
import pickle
import random
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

from evaluation.evaluation import eval_edge_prediction, eval_node_classification, \
    eval_tensor_edge_prediction, eval_tensor_node_classification, \
    eval_cold_node_classification, eval_tensor_node_classification_v3
from model.tgn import TGN, TensorTGN, SupervisedTGN, SupervisedTensorTGN
from utils.utils import EarlyStopMonitor, TRandEdgeSampler, TensorRandEdgeSampler, get_neighbor_finder
from utils.data_processing import compute_time_statistics, Data

from data_processing.read_tweet import get_dataset, get_dataset_cold, dict_to_df, \
    get_netease_dataset, get_netease_dataset_cold

def load_and_process(different_new_nodes_between_val_and_test=False, reindex=False, one_hot=True, elist=[0, 1], platform='local', is_cold=False):
    if not is_cold:
        if platform=='local':
            data = get_dataset(filter_feature=True, positive_subgraph=True)
        else:
            data = get_netease_dataset(filter_feature=True, positive_subgraph=True)
        weak_edges, strong_edges, features, labels = [dict_to_df(item) for item in data]
        edge_list = [strong_edges, weak_edges]
    else:
        if platform=='local':
            data = get_dataset_cold(filter_feature=True, positive_subgraph=True, is_cold=is_cold)
        else:
            data = get_netease_dataset_cold(filter_feature=True, positive_subgraph=True)
        if not 2 in elist:
            elist += [2]
        weak_edges, strong_edges, cold_edges, features, labels = [dict_to_df(item) for item in data]
        edge_list = [strong_edges, weak_edges, cold_edges]
    edict = {e: edge_list[e] for e in elist}
    df = dict_to_df(edict, 'etype')
    x_col, y_col = 'role_id', 't_role_id'


    timestamps = np.sort(np.unique(df['timestamp']))
    etypes = np.unique(df['etype'])
    nodes = np.unique(df[[x_col, y_col]])
    n_classes = len(np.unique(labels['label']))

    if different_new_nodes_between_val_and_test:
        val_time, test_time = int(len(timestamps) * 0.7), int(len(timestamps) * 0.8)
    else:
        val_time, test_time = int(len(timestamps) * 0.9), int(len(timestamps) * 0.9)
    val_time, test_time = timestamps[val_time], timestamps[test_time]
    print(f'timestamps split {val_time}, {test_time}')

    if reindex:
        node_to_index = {node: index+1 for index, node in enumerate(nodes)}
        df.loc[:, x_col] = df[x_col].map(node_to_index)
        df.loc[:, y_col] = df[y_col].map(node_to_index)
        features.index = features.index.map(node_to_index)
        labels.index = labels.index.map(node_to_index)

    else:
        node_to_index = {node: node for node in nodes}
    with open('inference/training_node2index.pkl', 'wb') as f:
        pickle.dump(node_to_index, f)

    n_node = max(node_to_index.values()) + 1
    features.index.name = 'role_id'
    labels.index.name = 'role_id'
    features = features.reset_index()
    labels = labels.reset_index()
    features = features.set_index(['role_id', 'timestamp'])
    labels = labels.set_index(['role_id', 'timestamp'])

    # for performance consideration
    features = features.sort_index()
    labels = labels.sort_index()

    # for no neighbor
    for ts in timestamps:
        features.loc[(0, ts), :] = np.zeros(features.shape[1])
        labels.loc[(0, ts), :] = np.zeros(labels.shape[1])

    # features.loc[(0, 0), :] = np.zeros(features.shape[1])
    # labels.loc[(0, 0), :] = np.zeros(labels.shape[1])
    zero_line = pd.DataFrame(np.zeros([1, df.shape[1]]), columns=df.columns)
    df = pd.concat([zero_line, df], axis=0)

    # one hot
    if one_hot:
        one_hot = np.eye(n_classes)
        labels = pd.DataFrame(one_hot[labels.values.squeeze().astype(int)].squeeze(),
                              index=labels.index, columns=[0, 1])

    # assert df.values.max() == len(node_to_index) - 1

    '''debug: random mask feature'''
    # feat_dim = features.shape[1]
    # idx_list = list(range(feat_dim))
    # random.shuffle(idx_list)
    # col_list = features.columns[idx_list[:int(feat_dim*0.25)]]
    # features = features[col_list]

    '''!!!important: ensure no information leakage!!!'''
    df = df.sort_values('timestamp')

    sources = df[x_col].values.astype(np.int64)
    destinations = df[y_col].values.astype(np.int64)
    edge_idxs = np.arange(df.shape[0])
    timestamps = df['timestamp'].values.astype(np.int64)
    edge_features = df['etype'].values.reshape([-1, 1])
    source_labels = labels.loc[[(s, t) for s, t in zip(sources, timestamps)]].values.astype(np.int64)
    dst_labels = labels.loc[[(s, t) for s, t in zip(destinations, timestamps)]].values.astype(np.int64)
    # for inference
    index_to_node = {v: k for k,v in node_to_index.items()}
    source_ids = df[x_col].map(index_to_node).values.astype(np.int64)
    destination_ids = df[y_col].map(index_to_node).values.astype(np.int64)

    full_data = Data(sources, destinations, timestamps, edge_idxs, source_labels, dst_labels, src_ids=source_ids, dst_ids=destination_ids)

    random.seed(2020)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # Compute nodes which appear at test time
    test_node_set = set(sources[timestamps > val_time]).union(
        set(destinations[timestamps > val_time]))
    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training
    # new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * len(test_node_set))))

    # Mask saying for each source and destination whether they are new test nodes
    new_test_source_mask = df[x_col].map(lambda x: x in new_test_node_set).values.astype(bool)
    new_test_destination_mask = df[y_col].map(lambda x: x in new_test_node_set).values.astype(bool)

    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # For train we keep edges happening before the validation time which do not involve any new node
    # used for inductiveness
    train_mask = np.logical_and(timestamps < val_time, observed_edges_mask)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], source_labels[train_mask], dst_labels[train_mask],
                      src_ids=source_ids[train_mask], dst_ids=destination_ids[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.sources).union(train_data.destinations)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(timestamps < test_time, timestamps >= val_time)
    test_mask = timestamps >= test_time

    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

        edge_contains_new_val_node_mask = np.array(
            [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array(
            [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


    else:
        edge_contains_new_node_mask = np.array(
            [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], source_labels[val_mask], dst_labels[val_mask],
                    src_ids=source_ids[val_mask], dst_ids=destination_ids[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], source_labels[test_mask], dst_labels[test_mask],
                     src_ids=source_ids[test_mask], dst_ids=destination_ids[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask], edge_idxs[new_node_val_mask],
                             source_labels[new_node_val_mask], dst_labels[new_node_val_mask],
                             src_ids=source_ids[new_node_val_mask], dst_ids=destination_ids[new_node_val_mask])

    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              source_labels[new_node_test_mask], dst_labels[new_node_test_mask],
                              src_ids=source_ids[new_node_test_mask], dst_ids=destination_ids[new_node_test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
        len(new_test_node_set)))

    return features, source_labels, dst_labels, edge_features, full_data, train_data, val_data, test_data, \
        new_node_val_data, new_node_test_data, n_node, node_to_index, n_classes



def supervised_main(batch_size=None):
    torch.manual_seed(0)
    np.random.seed(0)

    ### Argument and global variables
    parser = argparse.ArgumentParser('TGN self-supervised training')
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                        default='wikipedia')
    parser.add_argument('--bs', type=int, default=200, help='Batch_size')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                      'backprop')
    parser.add_argument('--use_memory', action='store_true',
                        help='Whether to augment the model with a node memory')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
        "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
        "mlp", "identity"], help='Type of message function')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=[
        "gru", "rnn"], help='Type of memory updater')
    parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                       'aggregator')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                    'each user')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--randomize_features', action='store_true',
                        help='Whether to randomize node features')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--dyrep', action='store_true',
                        help='Whether to run the dyrep model')
    parser.add_argument('--edges', type=str, default='0,1', help='Which graph to use')
    parser.add_argument('--no_time', action='store_true', help='Whether to consider time')

    args, _ = parser.parse_known_args()

    if batch_size is not None:
        BATCH_SIZE = batch_size
    else:
        BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    USE_MEMORY = args.use_memory
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    EDGES = list(map(int, args.edges.split(',')))
    NO_TIME = args.no_time
    if NO_TIME:
        USE_MEMORY = False

    filepath = os.path.dirname(os.path.abspath(__file__))
    Path(f"{filepath}/saved_models/").mkdir(parents=True, exist_ok=True)
    Path(f"{filepath}/saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f'{filepath}/saved_models/{args.prefix}-{args.data}.pth'
    get_checkpoint_path = lambda \
            epoch: f'{filepath}/saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
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
    logger.info('######TGN without Tensor!!!######')
    logger.info(f'batch size: {BATCH_SIZE}')

    ### Extract data for training, validation and testing
    # node_features, src_labels, dst_labels, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
    #     new_node_test_data, n_node, node_to_index, n_classes = load_and_process(True, reindex=True, elist=EDGES)
    node_features, src_labels, dst_labels, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
        new_node_test_data, n_node, node_to_index, n_classes = load_and_process(True, reindex=True, elist=EDGES, platform='local')

    torch.cuda.empty_cache()

    # Initialize training neighbor finder to retrieve temporal graph
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
    # across different runs
    # NB: in the inductive setting, negatives are sampled only amongst other new nodes
    train_rand_sampler = TRandEdgeSampler(train_data.sources, train_data.destinations, train_data.timestamps)
    val_rand_sampler = TRandEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps, seed=0)
    nn_val_rand_sampler = TRandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                           new_node_val_data.timestamps,
                                           seed=1)
    test_rand_sampler = TRandEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps, seed=2)
    nn_test_rand_sampler = TRandEdgeSampler(new_node_test_data.sources,
                                            new_node_test_data.destinations, new_node_test_data.timestamps,
                                            seed=3)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    for i in range(args.n_runs):
        results_path = "{}/results/{}_{}.pkl".format(filepath, args.prefix, i) if i > 0 else "{}/results/{}.pkl".format(filepath, args.prefix)
        Path(f"{filepath}/results/").mkdir(parents=True, exist_ok=True)

        # Initialize Model
        tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                  edge_features=edge_features, device=device, n_node=n_node,
                  n_layers=NUM_LAYER,
                  n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                  message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                  memory_update_at_start=not args.memory_update_at_end,
                  embedding_module_type=args.embedding_module,
                  message_function=args.message_function,
                  aggregator_type=args.aggregator,
                  memory_updater_type=args.memory_updater,
                  n_neighbors=NUM_NEIGHBORS,
                  mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                  mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message,
                  dyrep=args.dyrep, no_time=NO_TIME)

        model = SupervisedTGN(tgn, n_classes)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model = model.to(device)

        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / BATCH_SIZE)

        logger.info('num of training instances: {}'.format(num_instance))
        logger.info('num of batches per epoch: {}'.format(num_batch))
        idx_list = np.arange(num_instance)

        new_nodes_val_aps = []
        val_aps = []
        new_nodes_val_aucs = []
        val_node_aucs = []
        new_nodes_val_cls_aps = []
        val_cls_aps = []
        epoch_times = []
        total_epoch_times = []
        train_losses = []

        early_stopper = EarlyStopMonitor(max_round=args.patience)
        for epoch in range(NUM_EPOCH):
            start_epoch = time.time()
            ### Training

            # Reinitialize memory of the model at the start of each epoch
            if USE_MEMORY:
                model.tgn.memory.__init_memory__()

            # Train using only training graph
            model.tgn.set_neighbor_finder(train_ngh_finder)
            m_loss = []

            logger.info('start {} epoch'.format(epoch))
            for k in range(0, num_batch, args.backprop_every):
                loss = 0
                optimizer.zero_grad()

                # Custom loop to allow to perform backpropagation only every a certain number of batches
                for j in range(args.backprop_every):
                    batch_idx = k + j

                    if batch_idx >= num_batch:
                        continue

                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = min(num_instance, start_idx + BATCH_SIZE)
                    sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                        train_data.destinations[start_idx:end_idx]
                    edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                    timestamps_batch = train_data.timestamps[start_idx:end_idx]
                    labels_batch = train_data.labels[start_idx:end_idx]
                    dst_labels_batch = train_data.dst_labels[start_idx:end_idx]

                    size = len(sources_batch)
                    # _, negatives_batch = train_rand_sampler.sample(size)
                    _, negatives_batch = train_rand_sampler.sample(timestamps_batch)

                    # with torch.no_grad():
                    #     pos_label = torch.ones(size, dtype=torch.float, device=device)
                    #     neg_label = torch.zeros(size, dtype=torch.float, device=device)

                    model = model.train()

                    loss = model(sources_batch, destinations_batch, negatives_batch, timestamps_batch,
                                 edge_idxs_batch, labels_batch, dst_labels_batch, NUM_NEIGHBORS)

                loss /= args.backprop_every

                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())

                # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
                # the start of time
                if USE_MEMORY:
                    model.tgn.memory.detach_memory()

            epoch_time = time.time() - start_epoch
            epoch_times.append(epoch_time)

            ### Validation
            # Validation uses the full graph
            model.tgn.set_neighbor_finder(full_ngh_finder)

            if USE_MEMORY:
                # Backup memory at the end of training, so later we can restore it and use it for the
                # validation on unseen nodes
                train_memory_backup = model.tgn.memory.backup_memory()

            val_ap, val_auc, val_1 = eval_edge_prediction(model=model.tgn,
                                                   negative_edge_sampler=val_rand_sampler,
                                                   data=val_data,
                                                   n_neighbors=NUM_NEIGHBORS)
            val_node_auc, val_node_ap, val_node_1 = eval_node_classification(model, val_data, full_data.edge_idxs, BATCH_SIZE,
                                               n_neighbors=NUM_NEIGHBORS)

            if USE_MEMORY:
                val_memory_backup = model.tgn.memory.backup_memory()
                # Restore memory we had at the end of training to be used when validating on new nodes.
                # Also backup memory after validation so it can be used for testing (since test edges are
                # strictly later in time than validation edges)
                model.tgn.memory.restore_memory(train_memory_backup)

            # Validate on unseen nodes
            nn_val_ap, nn_val_auc, nn_val_1 = eval_edge_prediction(model=tgn,
                                                         negative_edge_sampler=val_rand_sampler,
                                                         data=new_node_val_data,
                                                         n_neighbors=NUM_NEIGHBORS)
            nn_val_node_auc, nn_val_node_ap, nn_val_node_1 = eval_node_classification(model, new_node_val_data, full_data.edge_idxs, BATCH_SIZE,
                                               n_neighbors=NUM_NEIGHBORS)

            if USE_MEMORY:
                # Restore memory we had at the end of validation
                # model.tgn.memory.restore_memory(val_memory_backup)
                model.tgn.memory.restore_memory(train_memory_backup)

            new_nodes_val_aps.append(nn_val_ap)
            val_aps.append(val_ap)
            new_nodes_val_aucs.append(nn_val_node_auc)
            val_node_aucs.append(val_node_auc)
            new_nodes_val_cls_aps.append(nn_val_node_ap)
            val_cls_aps.append(val_node_ap)
            train_losses.append(np.mean(m_loss))

            # Save temporary results to disk
            pickle.dump({
                "val_aps": val_aps,
                "new_nodes_val_aps": new_nodes_val_aps,
                "train_losses": train_losses,
                "epoch_times": epoch_times,
                "total_epoch_times": total_epoch_times
            }, open(results_path, "wb"))

            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)

            logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
            logger.info(
                'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
            logger.info(
                'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
            logger.info(
                'val 1: {}, new node val 1: {}'.format(val_1, nn_val_1))

            logger.info(
                'val classification auc: {}, new node val auc: {}'.format(val_node_auc, nn_val_node_auc))
            logger.info(
                'val classification ap: {}, new node val ap: {}'.format(val_node_ap, nn_val_node_ap))
            logger.info(
                'val classification 1: {}, new node val 1: {}'.format(val_node_1, nn_val_node_1))

            # Early stopping
            if early_stopper.early_stop_check(nn_val_node_auc):
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                best_model_path = get_checkpoint_path(early_stopper.best_epoch)
                model.load_state_dict(torch.load(best_model_path))
                logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                model.eval()
                break
            else:
                torch.save(model.state_dict(), get_checkpoint_path(epoch))

        # Training has finished, we have loaded the best model, and we want to backup its current
        # memory (which has seen validation edges) so that it can also be used when testing on unseen
        # nodes
        if USE_MEMORY:
            val_memory_backup = model.tgn.memory.backup_memory()

        ### Test
        model.tgn.embedding_module.neighbor_finder = full_ngh_finder
        test_ap, test_auc, test_1 = eval_edge_prediction(model=model.tgn,
                                                 negative_edge_sampler=test_rand_sampler,
                                                 data=test_data,
                                                 n_neighbors=NUM_NEIGHBORS)

        if USE_MEMORY:
            model.tgn.memory.restore_memory(val_memory_backup)

        # Test on unseen nodes
        nn_test_ap, nn_test_auc, nn_test_1 = eval_edge_prediction(model=model.tgn,
                                                       negative_edge_sampler=nn_test_rand_sampler,
                                                       data=new_node_test_data,
                                                       n_neighbors=NUM_NEIGHBORS)

        logger.info(
            'Test statistics: Old nodes -- auc: {}, ap: {}, 1: {}'.format(test_auc, test_ap, test_1))
        logger.info(
            'Test statistics: New nodes -- auc: {}, ap: {}, 1: {}'.format(nn_test_auc, nn_test_ap,nn_test_1))

        if USE_MEMORY:
            model.tgn.memory.restore_memory(val_memory_backup)

        test_node_auc, test_node_ap, test_node_1 = eval_node_classification(model, test_data, full_data.edge_idxs,
                                                                            BATCH_SIZE,
                                                                            n_neighbors=NUM_NEIGHBORS)

        if USE_MEMORY:
            model.tgn.memory.restore_memory(val_memory_backup)

        nn_test_node_auc, nn_test_node_ap, nn_test_node_1 = eval_node_classification(model, new_node_test_data,
                                                                                     full_data.edge_idxs, BATCH_SIZE,
                                                                                     n_neighbors=NUM_NEIGHBORS)
        logger.info(
            'test classification auc: {}, new node test auc: {}'.format(test_node_auc, nn_test_node_auc))
        logger.info(
            'test classification ap: {}, new node test ap: {}'.format(test_node_ap, nn_test_node_ap))
        logger.info(
            'test classification 1: {}, new node test 1: {}'.format(test_node_1, nn_test_node_1))


def supervised_inference():
    torch.manual_seed(0)
    np.random.seed(0)

    ### Argument and global variables
    parser = argparse.ArgumentParser('TGN self-supervised training')
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                        default='wikipedia')
    parser.add_argument('--bs', type=int, default=200, help='Batch_size')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                      'backprop')
    parser.add_argument('--use_memory', action='store_true',
                        help='Whether to augment the model with a node memory')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
        "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
        "mlp", "identity"], help='Type of message function')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=[
        "gru", "rnn"], help='Type of memory updater')
    parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                       'aggregator')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                    'each user')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--randomize_features', action='store_true',
                        help='Whether to randomize node features')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--dyrep', action='store_true',
                        help='Whether to run the dyrep model')
    parser.add_argument('--edges', type=str, default='0,1', help='Which graph to use')
    parser.add_argument('--no_time', action='store_true', help='Whether to consider time')
    parser.add_argument('--platform', type=str, default='local')

    args, _ = parser.parse_known_args()

    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    USE_MEMORY = args.use_memory
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    EDGES = list(map(int, args.edges.split(',')))
    NO_TIME = args.no_time
    if NO_TIME:
        USE_MEMORY = False

    filepath = os.path.dirname(os.path.abspath(__file__))
    Path(f"{filepath}/saved_models/").mkdir(parents=True, exist_ok=True)
    Path(f"{filepath}/saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f'{filepath}/saved_models/{args.prefix}-{args.data}.pth'
    get_checkpoint_path = lambda \
            epoch: f'{filepath}/saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
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

    ### Extract data for training, validation and testing
    node_features, src_labels, dst_labels, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
        new_node_test_data, n_node, node_to_index, n_classes = load_and_process(True, reindex=True, elist=EDGES,
                                                                                platform=args.platform)

    torch.cuda.empty_cache()

    # Initialize training neighbor finder to retrieve temporal graph
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
    # across different runs
    # NB: in the inductive setting, negatives are sampled only amongst other new nodes
    test_rand_sampler = TRandEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps, seed=2)
    nn_test_rand_sampler = TRandEdgeSampler(new_node_test_data.sources,
                                            new_node_test_data.destinations, new_node_test_data.timestamps,
                                            seed=3)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    for i in range(args.n_runs):
        results_path = "{}/results/{}_{}.pkl".format(filepath, args.prefix, i) if i > 0 else "{}/results/{}.pkl".format(
            filepath, args.prefix)
        Path(f"{filepath}/results/").mkdir(parents=True, exist_ok=True)

        # Initialize Model
        tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                  edge_features=edge_features, device=device, n_node=n_node,
                  n_layers=NUM_LAYER,
                  n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                  message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                  memory_update_at_start=not args.memory_update_at_end,
                  embedding_module_type=args.embedding_module,
                  message_function=args.message_function,
                  aggregator_type=args.aggregator,
                  memory_updater_type=args.memory_updater,
                  n_neighbors=NUM_NEIGHBORS,
                  mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                  mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message,
                  dyrep=args.dyrep, no_time=NO_TIME)

        model = SupervisedTGN(tgn, n_classes)
        # load checkpoint
        # best_epoch = 9
        # best_model_path = get_checkpoint_path(best_epoch)
        # model.load_state_dict(torch.load(best_model_path))
        # logger.info(f'Loaded the best model at epoch {best_epoch} for inference')
        # load model
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        logger.info(f'Loaded the best model for inference')
        model.eval()
        model = model.to(device)

        # Training has finished, we have loaded the best model, and we want to backup its current
        # memory (which has seen validation edges) so that it can also be used when testing on unseen
        # nodes
        if USE_MEMORY:
            val_memory_backup = model.tgn.memory.backup_memory()

        ### Test
        model.tgn.embedding_module.neighbor_finder = full_ngh_finder
        test_ap, test_auc, test_1 = eval_edge_prediction(model=model.tgn,
                                                         negative_edge_sampler=test_rand_sampler,
                                                         data=test_data,
                                                         n_neighbors=NUM_NEIGHBORS)

        if USE_MEMORY:
            model.tgn.memory.restore_memory(val_memory_backup)

        # Test on unseen nodes
        nn_test_ap, nn_test_auc, nn_test_1 = eval_edge_prediction(model=model.tgn,
                                                                  negative_edge_sampler=nn_test_rand_sampler,
                                                                  data=new_node_test_data,
                                                                  n_neighbors=NUM_NEIGHBORS)

        logger.info(
            'Test statistics: Old nodes -- auc: {}, ap: {}, 1: {}'.format(test_auc, test_ap, test_1))
        logger.info(
            'Test statistics: New nodes -- auc: {}, ap: {}, 1: {}'.format(nn_test_auc, nn_test_ap, nn_test_1))

        if USE_MEMORY:
            model.tgn.memory.restore_memory(val_memory_backup)

        test_node_auc, test_node_ap, test_node_1 = eval_node_classification(model, test_data, full_data.edge_idxs,
                                                                            BATCH_SIZE,
                                                                            n_neighbors=NUM_NEIGHBORS)

        if USE_MEMORY:
            model.tgn.memory.restore_memory(val_memory_backup)

        nn_test_node_auc, nn_test_node_ap, nn_test_node_1 = eval_node_classification(model, new_node_test_data,
                                                                                     full_data.edge_idxs, BATCH_SIZE,
                                                                                     n_neighbors=NUM_NEIGHBORS)
        logger.info(
            'test classification auc: {}, new node test auc: {}'.format(test_node_auc, nn_test_node_auc))
        logger.info(
            'test classification ap: {}, new node test ap: {}'.format(test_node_ap, nn_test_node_ap))
        logger.info(
            'test classification 1: {}, new node test 1: {}'.format(test_node_1, nn_test_node_1))


def tensor_supervised_main():
    torch.manual_seed(0)
    np.random.seed(0)

    ### Argument and global variables
    parser = argparse.ArgumentParser('TGN self-supervised training')
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                        default='wikipedia')
    parser.add_argument('--bs', type=int, default=200, help='Batch_size')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                      'backprop')
    parser.add_argument('--use_memory', action='store_true',
                        help='Whether to augment the model with a node memory')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
        "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
        "mlp", "identity"], help='Type of message function')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=[
        "gru", "rnn"], help='Type of memory updater')
    parser.add_argument('--aggregator', type=str, default="lastTensor", help='Type of message '
                                                                             'aggregator')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                    'each user')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--randomize_features', action='store_true',
                        help='Whether to randomize node features')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--dyrep', action='store_true',
                        help='Whether to run the dyrep model')
    parser.add_argument('--edges', type=str, default='0,1', help='Which graph to use')
    parser.add_argument('--no_time', action='store_true', help='Whether to consider time')
    parser.add_argument('--platform', type=str, default='local')
    parser.add_argument('--cold', action='store_true', help='Whether to add cold-start edges')

    args, _ = parser.parse_known_args()

    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    USE_MEMORY = args.use_memory
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    EDGES = list(map(int, args.edges.split(',')))
    NO_TIME = args.no_time
    if NO_TIME:
        USE_MEMORY = False

    filepath = os.path.dirname(os.path.abspath(__file__))
    Path(f"{filepath}/saved_models/").mkdir(parents=True, exist_ok=True)
    Path(f"{filepath}/saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f'{filepath}/saved_models/{args.prefix}-{args.data}-{args.cold}.pth'
    get_checkpoint_path = lambda \
            epoch: f'{filepath}/saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
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

    ### Extract data for training, validation and testing
    node_features, src_labels, dst_labels, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
        new_node_test_data, n_node, node_to_index, n_classes = load_and_process(True, reindex=True, elist=EDGES,
                                                                              platform=args.platform, is_cold=args.cold)

    torch.cuda.empty_cache()

    # Initialize training neighbor finder to retrieve temporal graph
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
    # across different runs
    # NB: in the inductive setting, negatives are sampled only amongst other new nodes
    train_rand_sampler = TRandEdgeSampler(train_data.sources, train_data.destinations, train_data.timestamps)
    val_rand_sampler = TRandEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps, seed=0)
    nn_val_rand_sampler = TRandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                           new_node_val_data.timestamps,
                                           seed=1)
    test_rand_sampler = TRandEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps, seed=2)
    nn_test_rand_sampler = TRandEdgeSampler(new_node_test_data.sources,
                                            new_node_test_data.destinations, new_node_test_data.timestamps,
                                            seed=3)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    for i in range(args.n_runs):
        results_path = "{}/results/{}_{}.pkl".format(filepath, args.prefix, i) if i > 0 else "{}/results/{}.pkl".format(
            filepath, args.prefix)
        Path(f"{filepath}/results/").mkdir(parents=True, exist_ok=True)

        # Initialize Model
        tgn = TensorTGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                        edge_features=edge_features, device=device, n_node=n_node,
                        n_layers=NUM_LAYER,
                        n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                        message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                        memory_update_at_start=not args.memory_update_at_end,
                        embedding_module_type=args.embedding_module,
                        message_function=args.message_function,
                        aggregator_type=args.aggregator,
                        memory_updater_type=args.memory_updater,
                        n_neighbors=NUM_NEIGHBORS,
                        mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                        mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                        use_source_embedding_in_message=args.use_source_embedding_in_message,
                        dyrep=args.dyrep, no_time=NO_TIME)

        model = SupervisedTensorTGN(tgn, n_classes)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model = model.to(device)

        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / BATCH_SIZE)

        logger.info('num of training instances: {}'.format(num_instance))
        logger.info('num of batches per epoch: {}'.format(num_batch))
        idx_list = np.arange(num_instance)

        new_nodes_val_aps = []
        val_aps = []
        new_nodes_val_aucs = []
        val_node_aucs = []
        new_nodes_val_cls_aps = []
        val_cls_aps = []
        epoch_times = []
        total_epoch_times = []
        train_losses = []

        early_stopper = EarlyStopMonitor(max_round=args.patience)
        for epoch in range(NUM_EPOCH):
            start_epoch = time.time()
            ### Training

            # Reinitialize memory of the model at the start of each epoch
            if USE_MEMORY:
                model.tgn.memory.__init_memory__()

            # Train using only training graph
            model.tgn.set_neighbor_finder(train_ngh_finder)
            m_loss = []

            logger.info('start {} epoch'.format(epoch))
            for k in tqdm(range(0, num_batch, args.backprop_every)):
                loss = 0
                optimizer.zero_grad()

                t1 = time.time()
                # Custom loop to allow to perform backpropagation only every a certain number of batches
                for j in range(args.backprop_every):
                    batch_idx = k + j

                    if batch_idx >= num_batch:
                        continue

                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = min(num_instance, start_idx + BATCH_SIZE)
                    sources_batch, destinations_batch = torch.from_numpy(train_data.sources[start_idx:end_idx]).to(
                        device), \
                        torch.from_numpy(train_data.destinations[start_idx:end_idx]).to(device)
                    edge_idxs_batch = torch.from_numpy(train_data.edge_idxs[start_idx: end_idx]).to(device)
                    timestamps_batch = torch.from_numpy(train_data.timestamps[start_idx:end_idx]).to(device)
                    labels_batch = torch.from_numpy(train_data.labels[start_idx:end_idx]).to(device)
                    dst_labels_batch = torch.from_numpy(train_data.dst_labels[start_idx:end_idx]).to(device)

                    size = len(sources_batch)
                    # _, negatives_batch = train_rand_sampler.sample(size)
                    _, negatives_batch = train_rand_sampler.sample(timestamps_batch)
                    negatives_batch = torch.from_numpy(negatives_batch).to(device)

                    # with torch.no_grad():
                    #     pos_label = torch.ones(size, dtype=torch.float, device=device)
                    #     neg_label = torch.zeros(size, dtype=torch.float, device=device)

                    model = model.train()

                    t2 = time.time()
                    loss = model(sources_batch, destinations_batch, negatives_batch, timestamps_batch,
                                 edge_idxs_batch, labels_batch, dst_labels_batch, NUM_NEIGHBORS)

                t3 = time.time()
                # print(f"\rOne batch takes {t3 - t1}, during data takes {t2-t1}, model takes {t3-t2}", end='')
                # sys.stdout.write(f"One batch takes {t3 - t1}, during data takes {t2-t1}, model takes {t3-t2}")
                # sys.stdout.flush()
                loss /= args.backprop_every

                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())

                # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
                # the start of time
                if USE_MEMORY:
                    model.tgn.memory.detach_memory()
            if k >= num_batch // 2:
                epoch_time = time.time() - start_epoch
                epoch_times.append(epoch_time)

                ### Validation
                # Validation uses the full graph
                model.tgn.set_neighbor_finder(full_ngh_finder)

                if USE_MEMORY:
                    # Backup memory at the end of training, so later we can restore it and use it for the
                    # validation on unseen nodes
                    train_memory_backup = model.tgn.memory.backup_memory()
                val_ap, val_auc, val_1 = eval_tensor_edge_prediction(model=model.tgn,
                                                                     negative_edge_sampler=val_rand_sampler,
                                                                     data=val_data,
                                                                     n_neighbors=NUM_NEIGHBORS,
                                                                     batch_size=2 * BATCH_SIZE)
                if USE_MEMORY:
                    # val_memory_backup = model.tgn.memory.backup_memory()
                    # Restore memory we had at the end of training to be used when validating on new nodes.
                    # Also backup memory after validation so it can be used for testing (since test edges are
                    # strictly later in time than validation edges)
                    model.tgn.memory.restore_memory(train_memory_backup)
                val_node_auc, val_node_ap, val_node_acc, val_node_1 = eval_tensor_node_classification(model, val_data,
                                                                                                      full_data.edge_idxs,
                                                                                                      BATCH_SIZE,
                                                                                                      n_neighbors=NUM_NEIGHBORS)

                # if USE_MEMORY:
                #     val_memory_backup = model.tgn.memory.backup_memory()
                #     # Restore memory we had at the end of training to be used when validating on new nodes.
                #     # Also backup memory after validation so it can be used for testing (since test edges are
                #     # strictly later in time than validation edges)
                #     model.tgn.memory.restore_memory(train_memory_backup)

                # Validate on unseen nodes
                # nn_val_ap, nn_val_auc, nn_val_1 = eval_tensor_edge_prediction(model=tgn,
                #                                              negative_edge_sampler=val_rand_sampler,
                #                                              data=new_node_val_data,
                #                                              n_neighbors=NUM_NEIGHBORS,
                #                                              batch_size=2*BATCH_SIZE)
                # nn_val_node_auc, nn_val_node_ap, nn_val_node_1 = eval_tensor_node_classification(model, new_node_val_data, full_data.edge_idxs, 2*BATCH_SIZE,
                #                                    n_neighbors=NUM_NEIGHBORS)

                if USE_MEMORY:
                    # Restore memory we had at the end of validation
                    model.tgn.memory.restore_memory(train_memory_backup)

                # new_nodes_val_aps.append(nn_val_ap)
                val_aps.append(val_ap)
                # new_nodes_val_aucs.append(nn_val_node_auc)
                val_node_aucs.append(val_node_auc)
                # new_nodes_val_cls_aps.append(nn_val_node_ap)
                val_cls_aps.append(val_node_ap)
                train_losses.append(np.mean(m_loss))

                # Save temporary results to disk
                pickle.dump({
                    "val_aps": val_aps,
                    # "new_nodes_val_aps": new_nodes_val_aps,
                    "train_losses": train_losses,
                    "epoch_times": epoch_times,
                    "total_epoch_times": total_epoch_times
                }, open(results_path, "wb"))

                total_epoch_time = time.time() - start_epoch
                total_epoch_times.append(total_epoch_time)

                logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
                logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
                #                 logger.info(
                #                     'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
                #                 logger.info(
                #                     'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
                #                 logger.info(
                #                     'val 1: {}, new node val 1: {}'.format(val_1, nn_val_1))

                #                 logger.info(
                #                     'val classification auc: {}, new node val auc: {}'.format(val_node_auc, nn_val_node_auc))
                #                 logger.info(
                #                     'val classification ap: {}, new node val ap: {}'.format(val_node_ap, nn_val_node_ap))
                #                 logger.info(
                #                     'val classification 1: {}, new node val 1: {}'.format(val_node_1, nn_val_node_1))

                logger.info(
                    'val auc: {}, val ap: {}, val 1: {}'.format(val_auc, val_ap, val_1))
                logger.info(
                    'val classification auc: {}, val classification ap: {}, val classification acc: {}, val classification 1: {}'.format(
                        val_node_auc, val_node_ap, val_node_acc, val_node_1))

                # Early stopping
                if early_stopper.early_stop_check(val_node_auc):
                    logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                    logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                    best_model_path = get_checkpoint_path(early_stopper.best_epoch)
                    model.load_state_dict(torch.load(best_model_path))
                    logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                    model.eval()
                    break
                else:
                    torch.save(model.state_dict(), get_checkpoint_path(epoch))

        # Training has finished, we have loaded the best model, and we want to backup its current
        # memory (which has seen validation edges) so that it can also be used when testing on unseen
        # nodes
        if USE_MEMORY:
            train_memory_backup = model.tgn.memory.backup_memory()

        ### Test
        model.tgn.embedding_module.neighbor_finder = full_ngh_finder
        test_ap, test_auc, test_1 = eval_tensor_edge_prediction(model=model.tgn,
                                                                negative_edge_sampler=test_rand_sampler,
                                                                data=test_data,
                                                                n_neighbors=NUM_NEIGHBORS,
                                                                batch_size=2 * BATCH_SIZE)

        # if USE_MEMORY:
        #     model.tgn.memory.restore_memory(train_memory_backup)

        # # Test on unseen nodes
        # nn_test_ap, nn_test_auc, nn_test_1 = eval_tensor_edge_prediction(model=model.tgn,
        #                                                negative_edge_sampler=nn_test_rand_sampler,
        #                                                data=new_node_test_data,
        #                                                n_neighbors=NUM_NEIGHBORS,
        #                                                batch_size=2*BATCH_SIZE)

        logger.info(
            'Test statistics: Old nodes -- auc: {}, ap: {}, 1: {}'.format(test_auc, test_ap, test_1))
        # logger.info(
        #     'Test statistics: New nodes -- auc: {}, ap: {}, 1: {}'.format(nn_test_auc, nn_test_ap,nn_test_1))

        if USE_MEMORY:
            model.tgn.memory.restore_memory(train_memory_backup)

        test_node_auc, test_node_ap, test_node_acc, test_node_1 = eval_tensor_node_classification(model, test_data,
                                                                                                  full_data.edge_idxs,
                                                                                                  2 * BATCH_SIZE,
                                                                                                  n_neighbors=NUM_NEIGHBORS)

        # if USE_MEMORY:
        #     model.tgn.memory.restore_memory(train_memory_backup)

        # nn_test_node_auc, nn_test_node_ap, nn_test_node_1 = eval_tensor_node_classification(model, new_node_test_data,
        #                                                                              full_data.edge_idxs, 2*BATCH_SIZE,
        #                                                                              n_neighbors=NUM_NEIGHBORS)
        # logger.info(
        #     'test classification auc: {}, new node test auc: {}'.format(test_node_auc, nn_test_node_auc))
        # logger.info(
        #     'test classification ap: {}, new node test ap: {}'.format(test_node_ap, nn_test_node_ap))
        # logger.info(
        #     'test classification 1: {}, new node test 1: {}'.format(test_node_1, nn_test_node_1))

        logger.info(
            'test classification auc: {}, test classification ap: {}, test classification acc: {}, test classification 1: {}'.format(
                test_node_auc, test_node_ap, test_node_acc, test_node_1))

        # Save results for this run
        pickle.dump({
            "val_aps": val_aps,
            # "new_nodes_val_aps": new_nodes_val_aps,
            "test_ap": test_ap,
            # "new_node_test_ap": nn_test_ap,
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        logger.info('Saving TGN model')
        if USE_MEMORY:
            # Restore memory at the end of validation (save a model which is ready for testing)
            model.tgn.memory.restore_memory(train_memory_backup)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        logger.info('TGN model saved')


def tensor_supervised_inference():
    torch.manual_seed(0)
    np.random.seed(0)

    ### Argument and global variables
    parser = argparse.ArgumentParser('TGN self-supervised training')
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                        default='wikipedia')
    parser.add_argument('--bs', type=int, default=200, help='Batch_size')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                      'backprop')
    parser.add_argument('--use_memory', action='store_true',
                        help='Whether to augment the model with a node memory')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
        "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
        "mlp", "identity"], help='Type of message function')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=[
        "gru", "rnn"], help='Type of memory updater')
    parser.add_argument('--aggregator', type=str, default="lastTensor", help='Type of message '
                                                                             'aggregator')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                    'each user')
    parser.add_argument('--different_new_nodes', action='store_true',
                        help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--randomize_features', action='store_true',
                        help='Whether to randomize node features')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true',
                        help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--dyrep', action='store_true',
                        help='Whether to run the dyrep model')
    parser.add_argument('--edges', type=str, default='0,1', help='Which graph to use')
    parser.add_argument('--no_time', action='store_true', help='Whether to consider time')
    parser.add_argument('--platform', type=str, default='local')
    parser.add_argument('--cold', action='store_true', help='Whether to add cold-start edges')

    args, _ = parser.parse_known_args()

    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    USE_MEMORY = args.use_memory
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    EDGES = list(map(int, args.edges.split(',')))
    NO_TIME = args.no_time
    if NO_TIME:
        USE_MEMORY = False

    filepath = os.path.dirname(os.path.abspath(__file__))
    Path(f"{filepath}/saved_models/").mkdir(parents=True, exist_ok=True)
    Path(f"{filepath}/saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f'{filepath}/saved_models/{args.prefix}-{args.data}-{args.cold}.pth'
    get_checkpoint_path = lambda \
            epoch: f'{filepath}/saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
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

    ### Extract data for training, validation and testing
    node_features, src_labels, dst_labels, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
        new_node_test_data, n_node, node_to_index, n_classes = load_and_process(True, reindex=True, elist=EDGES,
                                                                              platform=args.platform, is_cold=args.cold)

    torch.cuda.empty_cache()

    # Initialize training neighbor finder to retrieve temporal graph
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
    # across different runs
    # NB: in the inductive setting, negatives are sampled only amongst other new nodes
    test_rand_sampler = TRandEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps, seed=2)
    nn_test_rand_sampler = TRandEdgeSampler(new_node_test_data.sources,
                                            new_node_test_data.destinations, new_node_test_data.timestamps,
                                            seed=3)

    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    for i in range(args.n_runs):
        results_path = "{}/results/{}_{}.pkl".format(filepath, args.prefix, i) if i > 0 else "{}/results/{}.pkl".format(
            filepath, args.prefix)
        Path(f"{filepath}/results/").mkdir(parents=True, exist_ok=True)

        # Initialize Model
        tgn = TensorTGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                        edge_features=edge_features, device=device, n_node=n_node,
                        n_layers=NUM_LAYER,
                        n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                        message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                        memory_update_at_start=not args.memory_update_at_end,
                        embedding_module_type=args.embedding_module,
                        message_function=args.message_function,
                        aggregator_type=args.aggregator,
                        memory_updater_type=args.memory_updater,
                        n_neighbors=NUM_NEIGHBORS,
                        mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                        mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                        use_source_embedding_in_message=args.use_source_embedding_in_message,
                        dyrep=args.dyrep, no_time=NO_TIME)

        model = SupervisedTensorTGN(tgn, n_classes)
        # load checkpoint
        # best_epoch = 9
        # best_model_path = get_checkpoint_path(best_epoch)
        # model.load_state_dict(torch.load(best_model_path))
        # logger.info(f'Loaded the best model at epoch {best_epoch} for inference')
        # load model
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        logger.info(f'Loaded the best model for inference')
        model.eval()
        model = model.to(device)

        # Training has finished, we have loaded the best model, and we want to backup its current
        # memory (which has seen validation edges) so that it can also be used when testing on unseen
        # nodes
        if USE_MEMORY:
            val_memory_backup = model.tgn.memory.backup_memory()

        ### Test
        model.tgn.embedding_module.neighbor_finder = full_ngh_finder
        # test_ap, test_auc, test_1 = eval_tensor_edge_prediction(model=model.tgn,
        #                                                  negative_edge_sampler=test_rand_sampler,
        #                                                  data=test_data,
        #                                                  n_neighbors=NUM_NEIGHBORS,
        #                                                  batch_size=BATCH_SIZE*2)

        # if USE_MEMORY:
        #     model.tgn.memory.restore_memory(val_memory_backup)
        # Test on unseen nodes
        # nn_test_ap, nn_test_auc, nn_test_1 = eval_tensor_edge_prediction(model=model.tgn,
        #                                                           negative_edge_sampler=nn_test_rand_sampler,
        #                                                           data=new_node_test_data,
        #                                                           n_neighbors=NUM_NEIGHBORS,
        #                                                           batch_size=BATCH_SIZE*2)

        # logger.info(
        #     'Test statistics: Old nodes -- auc: {}, ap: {}, 1: {}'.format(test_auc, test_ap, test_1))
        # logger.info(
        #     'Test statistics: New nodes -- auc: {}, ap: {}, 1: {}'.format(nn_test_auc, nn_test_ap, nn_test_1))

        if USE_MEMORY:
            model.tgn.memory.restore_memory(val_memory_backup)

        test_node_auc, test_node_ap, test_node_acc, test_node_1 = eval_tensor_node_classification(model, test_data,
                                                                                                  full_data.edge_idxs,
                                                                                                  2 * BATCH_SIZE,
                                                                                                  n_neighbors=NUM_NEIGHBORS)

        if USE_MEMORY:
            model.tgn.memory.restore_memory(val_memory_backup)

        nn_test_node_auc, nn_test_node_ap, nn_test_node_acc, nn_test_node_1 = eval_tensor_node_classification(model,
                                                                                                              new_node_test_data,
                                                                                                              full_data.edge_idxs,
                                                                                                              2 * BATCH_SIZE,
                                                                                                              n_neighbors=NUM_NEIGHBORS)
        logger.info(
            'test classification auc: {}, new node test auc: {}'.format(test_node_auc, nn_test_node_auc))
        logger.info(
            'test classification ap: {}, new node test ap: {}'.format(test_node_ap, nn_test_node_ap))
        logger.info(
            'test classification acc: {}, new node test acc: {}'.format(test_node_acc, nn_test_node_acc))
        logger.info(
            'test classification 1: {}, new node test 1: {}'.format(test_node_1, nn_test_node_1))
        # logger.info(
        #     'test classification auc: {}, test classification ap: {}, test classification 1: {}'.format(test_node_auc, test_node_ap, test_node_1))

        if USE_MEMORY:
            model.tgn.memory.restore_memory(val_memory_backup)

        test_node_auc, test_node_ap, test_node_acc, test_node_1 = eval_tensor_node_classification_v3(model, test_data,
                                                                                                     full_data.edge_idxs,
                                                                                                     2 * BATCH_SIZE,
                                                                                                     n_neighbors=NUM_NEIGHBORS)

        if USE_MEMORY:
            model.tgn.memory.restore_memory(val_memory_backup)

        nn_test_node_auc, nn_test_node_ap, nn_test_node_acc, nn_test_node_1 = eval_tensor_node_classification_v3(model,
                                                                                                                 new_node_test_data,
                                                                                                                 full_data.edge_idxs,
                                                                                                                 2 * BATCH_SIZE,
                                                                                                                 n_neighbors=NUM_NEIGHBORS)
        logger.info(
            'test node auc: {}, new node test auc: {}'.format(test_node_auc, nn_test_node_auc))
        logger.info(
            'test node ap: {}, new node test ap: {}'.format(test_node_ap, nn_test_node_ap))
        logger.info(
            'test node acc: {}, new node test acc: {}'.format(test_node_acc, nn_test_node_acc))
        logger.info(
            'test node 1: {}, new node test 1: {}'.format(test_node_1, nn_test_node_1))


if __name__ == '__main__':
    test_batch = [1024, 512, 256, 128, 64]
    for batch in test_batch[::-1]:
        supervised_main(batch)
    # supervised_inference()

    # tensor_supervised_main()
    # tensor_supervised_inference()
    
