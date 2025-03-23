import math, time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    if num_test_instance == 0:
        return 0, 0, 0

    val_lab, val_ap, val_auc = [], [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(timestamps_batch)

            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                  negative_samples, timestamps_batch,
                                                                  edge_idxs_batch, n_neighbors)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
            val_lab.append(true_label)

    val_lab = np.concatenate(val_lab)
    return np.mean(val_ap), np.mean(val_auc), sum(val_lab == 1) / len(val_lab)


def eval_tensor_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    device = model.device

    if num_test_instance == 0:
        return 0, 0, 0

    val_lab, val_ap, val_auc = [], [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory

        # for k in tqdm(range(num_test_batch)):
        for k in tqdm(range(1)):
            t0 = time.time()
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = torch.from_numpy(data.sources[s_idx:e_idx]).to(device)
            destinations_batch = torch.from_numpy(data.destinations[s_idx:e_idx]).to(device)
            timestamps_batch = torch.from_numpy(data.timestamps[s_idx:e_idx]).to(device)
            edge_idxs_batch = torch.from_numpy(data.edge_idxs[s_idx: e_idx]).to(device)

            t1 = time.time()
            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(timestamps_batch)
            negative_samples = torch.from_numpy(negative_samples).to(device)
            t2 = time.time()

            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                  negative_samples, timestamps_batch,
                                                                  edge_idxs_batch, n_neighbors)

            t3 = time.time()
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            # print(f"\rdata takes {t1-t0}, sampler takes {t2-t1}, model takes {t3-t2}, np takes {time.time()-t3}", end="")

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
            val_lab.append(true_label)

    val_lab = np.concatenate(val_lab)
    return np.mean(val_ap), np.mean(val_auc), sum(val_lab == 1) / len(val_lab)


# def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
#   pred_prob = np.zeros(len(data.sources))
#   num_instance = len(data.sources)
#   num_batch = math.ceil(num_instance / batch_size)
#
#   with torch.no_grad():
#     decoder.eval()
#     tgn.eval()
#     for k in range(num_batch):
#       s_idx = k * batch_size
#       e_idx = min(num_instance, s_idx + batch_size)
#
#       sources_batch = data.sources[s_idx: e_idx]
#       destinations_batch = data.destinations[s_idx: e_idx]
#       timestamps_batch = data.timestamps[s_idx:e_idx]
#       # edge_idxs_batch = edge_idxs[s_idx: e_idx]
#       edge_idxs_batch = data.edge_idxs[s_idx: e_idx]
#
#       source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
#                                                                                    destinations_batch,
#                                                                                    destinations_batch,
#                                                                                    timestamps_batch,
#                                                                                    edge_idxs_batch,
#                                                                                    n_neighbors)
#       pred_prob_batch = decoder(source_embedding).sigmoid()
#       pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()
#
#   auc_roc = roc_auc_score(data.labels, pred_prob)
#   return auc_roc

def eval_node_classification(model, data, edge_idxs, batch_size, n_neighbors):
    src_pred_prob = np.zeros(len(data.sources))
    dst_pred_prob = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    if num_instance == 0:
        return 0, 0, 0

    with torch.no_grad():
        model.tgn.eval()
        model.decoder.eval()

        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx: e_idx]
            destinations_batch = data.destinations[s_idx: e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            # edge_idxs_batch = edge_idxs[s_idx: e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            src_pred_prob_batch, dst_pred_prob_batch = \
                model.predict(sources_batch,
                              destinations_batch,
                              timestamps_batch,
                              edge_idxs_batch,
                              n_neighbors)

            src_pred_prob[s_idx: e_idx] = src_pred_prob_batch.argmax(dim=1).cpu().numpy()
            dst_pred_prob[s_idx: e_idx] = dst_pred_prob_batch.argmax(dim=1).cpu().numpy()

    pred_prob = np.concatenate([src_pred_prob, dst_pred_prob], axis=0)
    true_label = np.concatenate([data.labels, data.dst_labels], axis=0)
    auc_roc = roc_auc_score(true_label.argmax(1), pred_prob)
    ap = average_precision_score(true_label.argmax(1), pred_prob)
    true_1 = sum(true_label.argmax(1) == 1) / (num_instance * 2)
    return auc_roc, ap, true_1

def eval_cold_node_classification(model, data, edge_idxs, batch_size, n_neighbors):
  src_pred_prob = np.zeros(len(data.sources))
  dst_pred_prob = np.zeros(len(data.sources))
  src_cold_index = np.zeros(len(data.sources), dtype=bool)
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  if num_instance == 0:
    return 0, 0, 0

  with torch.no_grad():
    model.tgn.eval()
    model.decoder.eval()

    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      # edge_idxs_batch = edge_idxs[s_idx: e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]
      cold_batch = model.tgn.edge_raw_features[edge_idxs_batch] == 2

      src_pred_prob_batch, dst_pred_prob_batch = \
                    model.predict(sources_batch,
                             destinations_batch,
                             timestamps_batch,
                             edge_idxs_batch,
                             n_neighbors)

      src_pred_prob[s_idx: e_idx] = src_pred_prob_batch.argmax(dim=1).cpu().numpy()
      dst_pred_prob[s_idx: e_idx] = dst_pred_prob_batch.argmax(dim=1).cpu().numpy()
      src_cold_index[s_idx: e_idx] = cold_batch[:, 0].cpu().numpy()

  # pred_prob = np.concatenate([src_pred_prob[src_cold_index], dst_pred_prob[src_cold_index]], axis=0)
  # true_label = np.concatenate([data.labels[src_cold_index], data.dst_labels[src_cold_index]], axis=0)
  pred_prob = src_pred_prob[src_cold_index]
  true_label = data.labels[src_cold_index]
  if len(np.unique(true_label.argmax(1))) == len(np.unique(pred_prob)):
    auc_roc = roc_auc_score(true_label.argmax(1), pred_prob)
    ap = average_precision_score(true_label.argmax(1), pred_prob)
    true_1 = sum(true_label.argmax(1) == 1) / (len(pred_prob))
    return auc_roc, ap, true_1
  else:
    return 0.0, 0.0, 0.0

def eval_tensor_node_classification_v1(model, data, edge_idxs, batch_size, n_neighbors):
    src_pred_prob = np.zeros([len(data.sources), 2])
    dst_pred_prob = np.zeros([len(data.sources), 2])
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)
    device = model.device

    if num_instance == 0:
        return 0, 0, 0

    with torch.no_grad():
        model.tgn.eval()
        model.decoder.eval()

        for k in tqdm(range(num_batch)):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = torch.from_numpy(data.sources[s_idx: e_idx]).to(device)
            destinations_batch = torch.from_numpy(data.destinations[s_idx: e_idx]).to(device)
            timestamps_batch = torch.from_numpy(data.timestamps[s_idx:e_idx]).to(device)
            # edge_idxs_batch = edge_idxs[s_idx: e_idx]
            edge_idxs_batch = torch.from_numpy(data.edge_idxs[s_idx: e_idx]).to(device)

            src_pred_prob_batch, dst_pred_prob_batch = \
                model.predict(sources_batch,
                              destinations_batch,
                              timestamps_batch,
                              edge_idxs_batch,
                              n_neighbors)

            # src_pred_prob[s_idx: e_idx] = src_pred_prob_batch.argmax(dim=1).cpu().numpy()
            # dst_pred_prob[s_idx: e_idx] = dst_pred_prob_batch.argmax(dim=1).cpu().numpy()
            # src_pred_prob[s_idx: e_idx] = src_pred_prob_batch[:, 1].cpu().numpy()
            # dst_pred_prob[s_idx: e_idx] = dst_pred_prob_batch[:, 1].cpu().numpy()
            src_pred_prob[s_idx: e_idx] = src_pred_prob_batch.cpu().numpy()
            dst_pred_prob[s_idx: e_idx] = dst_pred_prob_batch.cpu().numpy()

    pred_prob = np.concatenate([src_pred_prob, dst_pred_prob], axis=0)
    true_label = np.concatenate([data.labels, data.dst_labels], axis=0)
    auc_roc = roc_auc_score(true_label.argmax(1), pred_prob[:, 1])
    ap = average_precision_score(true_label.argmax(1), pred_prob[:, 1])
    acc = sum(pred_prob.argmax(1) == true_label.argmax(1)) / len(true_label)
    true_1 = sum(true_label.argmax(1) == 1) / (num_instance * 2)
    return auc_roc, ap, acc, true_1


def eval_tensor_node_classification_v2(model, data, edge_idxs, batch_size, n_neighbors):
    src_pred_prob = np.zeros(len(data.sources))
    dst_pred_prob = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)
    device = model.device

    if num_instance == 0:
        return 0, 0, 0

    with torch.no_grad():
        model.tgn.eval()
        model.decoder.eval()

        for k in tqdm(range(num_batch)):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = torch.from_numpy(data.sources[s_idx: e_idx]).to(device)
            destinations_batch = torch.from_numpy(data.destinations[s_idx: e_idx]).to(device)
            timestamps_batch = torch.from_numpy(data.timestamps[s_idx:e_idx]).to(device)
            # edge_idxs_batch = edge_idxs[s_idx: e_idx]
            edge_idxs_batch = torch.from_numpy(data.edge_idxs[s_idx: e_idx]).to(device)

            src_pred_prob_batch, dst_pred_prob_batch = \
                model.predict(sources_batch,
                              destinations_batch,
                              timestamps_batch,
                              edge_idxs_batch,
                              n_neighbors)

            src_pred_prob[s_idx: e_idx] = src_pred_prob_batch.argmax(dim=1).cpu().numpy()
            dst_pred_prob[s_idx: e_idx] = dst_pred_prob_batch.argmax(dim=1).cpu().numpy()

    pred_prob = np.concatenate([data.src_ids[:, None], src_pred_prob[:, None]], axis=1)
    true_label = np.concatenate([data.src_ids[:, None], data.labels.argmax(1)[:, None]], axis=1)
    # print(pred_prob.shape, true_label.shape)
    # node-level
    pred_prob_df = pd.DataFrame(pred_prob, columns=['role_id', 'pred']).groupby('role_id').agg('max')
    true_label_df = pd.DataFrame(true_label, columns=['role_id', 'label']).drop_duplicates()
    # align
    concat_df = pd.concat([true_label_df, pred_prob_df], axis=1)
    concat_df = pd.merge(true_label_df, pred_prob_df, on='role_id', how='left')

    pred_prob = concat_df['pred'].values
    true_label = concat_df['label'].values
    auc_roc = roc_auc_score(true_label, pred_prob)
    ap = average_precision_score(true_label, pred_prob)
    true_1 = sum(true_label == 1) / (len(true_label))
    return auc_roc, ap, true_1


def eval_tensor_node_classification_v3(model, data, edge_idxs, batch_size, n_neighbors):
    device = model.device
    num_classes = model.n_classes
    src_pred_prob = np.zeros([len(data.sources), num_classes])
    dst_pred_prob = np.zeros([len(data.sources), num_classes])
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    if num_instance == 0:
        return 0, 0, 0

    with torch.no_grad():
        model.tgn.eval()
        model.decoder.eval()

        for k in tqdm(range(num_batch)):
            # for k in tqdm(range(1)):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = torch.from_numpy(data.sources[s_idx: e_idx]).to(device)
            destinations_batch = torch.from_numpy(data.destinations[s_idx: e_idx]).to(device)
            timestamps_batch = torch.from_numpy(data.timestamps[s_idx:e_idx]).to(device)
            # edge_idxs_batch = edge_idxs[s_idx: e_idx]
            edge_idxs_batch = torch.from_numpy(data.edge_idxs[s_idx: e_idx]).to(device)

            src_pred_prob_batch, dst_pred_prob_batch = \
                model.predict(sources_batch,
                              destinations_batch,
                              timestamps_batch,
                              edge_idxs_batch,
                              n_neighbors)

            src_pred_prob[s_idx: e_idx] = src_pred_prob_batch.softmax(1).cpu().numpy()
            dst_pred_prob[s_idx: e_idx] = dst_pred_prob_batch.softmax(1).cpu().numpy()

    all_nodes = np.concatenate([data.sources, data.destinations], axis=0)
    pred_prob = np.concatenate([src_pred_prob, dst_pred_prob], axis=0)
    true_label = np.concatenate([data.labels, data.dst_labels], axis=0)
    pred_score = pred_prob[:, 1]
    df = pd.DataFrame(np.concatenate([all_nodes.reshape(-1, 1), pred_prob.argmax(1).reshape(-1, 1),
                                      pred_score.reshape(-1, 1),
                                      true_label.argmax(1).reshape(-1, 1)], axis=1),
                      columns=['role_id', 'pred', 'pred_score', 'label'])
    df = df.drop_duplicates()

    #   ## check multiple pred for one node
    #   group_df = df.groupby('role_id').size()
    #   group_df1 = df[['role_id', 'pred', 'label']].groupby('role_id').size()
    #   print('duplicated role_id takes up:', group_df[group_df>1].shape[0]/group_df.shape[0], group_df1[group_df1>1].shape[0]/group_df1.shape[0])
    # #   assert group_df[group_df[group_df>1]].shape[0] == 0, 'multiple pred for one node'
    df = df.loc[df.groupby('role_id')['pred_score'].idxmax()]

    true_label = df['label'].values
    pred_prob = df['pred'].values
    pred_score = df['pred_score'].values
    auc_roc = roc_auc_score(true_label, pred_score)
    ap = average_precision_score(true_label, pred_score)
    acc = sum(true_label == pred_prob) / len(true_label)
    true_1 = sum(true_label == 1) / (len(true_label))
    return auc_roc, ap, acc, true_1


eval_tensor_node_classification = eval_tensor_node_classification_v1


def CHECKEQ(self, input_tensor):
    if (len(input_tensor)) == 1:
        return True
    # Expand dimensions and compare
    expanded_tensor = input_tensor.unsqueeze(1)
    comparison = expanded_tensor == expanded_tensor.transpose(0, 1)

    # Check for row-wise equality
    row_equality = comparison.all(-1)

    # Identify equivalent rows, ignoring self-comparison
    indices = torch.triu_indices(row_equality.size(0), row_equality.size(1), offset=1)
    equivalent_rows = row_equality[indices[0], indices[1]]

    return equivalent_rows.all()


def inference_node_classification(model, data, edge_idxs, batch_size, n_neighbors, save_str):
    device = model.device
    num_classes = model.n_classes
    src_pred_prob = np.zeros([len(data.sources), num_classes])
    dst_pred_prob = np.zeros([len(data.sources), num_classes])
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    if num_instance == 0:
        return 0, 0, 0

    with torch.no_grad():
        model.tgn.eval()
        model.decoder.eval()

        for k in tqdm(range(num_batch)):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = torch.from_numpy(data.sources[s_idx: e_idx]).to(device)
            destinations_batch = torch.from_numpy(data.destinations[s_idx: e_idx]).to(device)
            timestamps_batch = torch.from_numpy(data.timestamps[s_idx:e_idx]).to(device)
            # edge_idxs_batch = edge_idxs[s_idx: e_idx]
            edge_idxs_batch = torch.from_numpy(data.edge_idxs[s_idx: e_idx]).to(device)

            src_pred_prob_batch, dst_pred_prob_batch = \
                model.predict(sources_batch,
                              destinations_batch,
                              timestamps_batch,
                              edge_idxs_batch,
                              n_neighbors)
            # src_pred_prob_batch = src_pred_prob_batch[:, 1] * (src_pred_prob_batch[:, 0] < src_pred_prob_batch[:, 1])
            src_pred_prob[s_idx: e_idx] = src_pred_prob_batch.softmax(1).cpu().numpy()
            dst_pred_prob[s_idx: e_idx] = dst_pred_prob_batch.softmax(1).cpu().numpy()

    all_ids = np.concatenate([data.src_ids, data.dst_ids], axis=0)
    pred_prob = np.concatenate([src_pred_prob, dst_pred_prob], axis=0)
    pred_score = pred_prob[:, 1]
    pred_df = pd.DataFrame(np.concatenate([all_ids.reshape(-1, 1),
                                           pred_score.reshape(-1, 1)], axis=1),
                           columns=['role_id', 'value'])
    pred_df = pred_df.drop_duplicates()
    pred_df = pred_df.loc[pred_df.groupby('role_id')['value'].idxmax()]

    # pred_df = pd.DataFrame(np.concatenate([data.src_ids[:, None], src_pred_prob[:, None]], axis=1), columns=['role_id', 'value'])
    # pred_df = pred_df.groupby('role_id').agg('mean')
    pred_df.to_csv('%s_pred.csv' % (save_str))
    print('save prediction to %s' % ('%s_pred.csv' % (save_str)))