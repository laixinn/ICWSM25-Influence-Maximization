import copy, os, logging
from tqdm import tqdm
import torch
import torch.nn as nn
# from torchvision.ops import sigmoid_focal_loss

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from models.graphmae.utils import create_optimizer, accuracy

save_threshold = -1


def inference(model, loaders, num_classes, emb_dim, load_str, device='cuda', linear_prob=True, mute=False,
              save_str=None, supervised=False):
    # load model
    ckpt = torch.load('%s_best_model.ckpt' % (load_str))
    model.load_state_dict(ckpt['gnn_state_dict'])

    # encoder
    if not supervised:
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        encoder = LogisticRegression(emb_dim, num_classes)
    else:
        encoder = model.supervised_encoder

    model = model.to(device)
    model.eval()
    encoder = encoder.to(device)
    encoder.eval()

    x_out, y_out, id_out = [], [], []

    with torch.no_grad():
        for subgraph in tqdm(loaders):
            # embedding
            subgraph = subgraph[0].to(device)
            feat = subgraph.ndata["feat"]
            x = model.embed(subgraph, feat)
            # prediction
            y_pred = encoder(None, x)  #

            x_out.append(x.cpu().numpy())
            y_out.append(y_pred.softmax(1).cpu().numpy())
            id_out.append(subgraph.ndata['role_id'].cpu().numpy())
        x_out = np.concatenate(x_out, axis=0)
        y_out = np.concatenate(y_out, axis=0)
        id_out = np.concatenate(id_out, axis=0)

        # TODO: check argmax
        # y_out = y_out.argmax(axis=1)
        y_out = y_out[:, 1]  # * (y_out[:,0] < y_out[:,1])

    # numpy save
    # if not save_str is None:
    #     np.save('%s_emb.npy' % (save_str), x_out)
    #     np.save('%s_pred.npy' % (save_str), y_out)
    #     print('save embedding to %s and save prediction to %s' % \
    #           ('%s_emb.npy' % (save_str), '%s_pred.npy' % (save_str)))

    # csv save
    if not save_str is None:
        # emb_df = pd.DataFrame(np.concatenate([id_out[:,  None], x_out], axis=1), columns=['role_id']+list(range(x_out.shape[1])))
        # pred_df = pd.DataFrame(np.concatenate([id_out[:, None], y_out[:, None]], axis=1), columns=['role_id', 'value'])
        # emb_df.to_csv('%s_emb.csv' % (save_str))
        # pred_df.to_csv('%s_pred.csv' % (save_str))
        # print('save embedding to %s and save prediction to %s' % \
        #       ('%s_emb.csv' % (save_str), '%s_pred.csv' % (save_str)))
        pred_df = pd.DataFrame(np.concatenate([id_out.reshape(-1, 1), y_out.reshape(-1, 1)], axis=1),
                               columns=['role_id', 'value'])
        pred_df = pred_df.drop_duplicates()
        pred_df = pred_df.loc[pred_df.groupby('role_id')['value'].idxmax()]
        pred_df.to_csv('%s_pred.csv' % (save_str))
        print('save prediction to %s' % \
              ('%s_pred.csv' % (save_str)))


def supervised_evaluete(model, loaders, num_classes, lr_f=0.005, weight_decay_f=1e-5, max_epoch_f=100, device='cuda',
                        linear_prob=True, mute=False, save_str=None):
    global save_threshold

    with torch.no_grad():
        model.eval()
        val_out = []
        val_label = []
        for subgraph in tqdm(loaders[1]):
            subgraph = subgraph[0].to(device)
            val_pred = model.predict(subgraph, subgraph.ndata["feat"])
            val_out.append(val_pred)
            val_label.append(subgraph.ndata['label'])
        val_out = torch.cat(val_out, dim=0)
        val_label = torch.cat(val_label, dim=0)
        val_label = val_label.max(1)[1].type_as(val_label)

        val_acc = accuracy(val_out, val_label)
        val_ap = average_precision_score(val_label.cpu().numpy(), val_out[:, 1].cpu().numpy())
        val_auc = roc_auc_score(val_label.cpu().numpy(), val_out[:, 1].cpu().numpy())
        val_1 = (val_label == 1).sum() / len(val_label)

        test_out = []
        test_label = []
        for subgraph in tqdm(loaders[2]):
            subgraph = subgraph[0].to(device)
            test_pred = model.predict(subgraph, subgraph.ndata["feat"])
            test_out.append(test_pred)
            test_label.append(subgraph.ndata['label'])
        test_out = torch.cat(test_out, dim=0)
        test_label = torch.cat(test_label, dim=0)
        test_label = test_label.max(1)[1].type_as(test_label)

        test_acc = accuracy(test_out, test_label)
        test_ap = average_precision_score(test_label.cpu().numpy(), test_out[:, 1].cpu().numpy())
        test_auc = roc_auc_score(test_label.cpu().numpy(), test_out[:, 1].cpu().numpy())
        test_1 = (test_label == 1).sum() / len(test_label)

    if not mute:
        print(
            f"# val_1: {val_1: .4f}, val_acc:{val_acc: .4f}, val_ap:{val_ap: .4f}, val_auc:{val_auc: .4f}, "
            f"test_1: {test_1: .4f}, test_acc:{test_acc: .4f}, test_ap:{test_ap: .4f}, test_auc:{test_auc: .4f}")

    # save
    save_dict = {
        'encoder_state_dict': model.state_dict(),
        # 'best_val_epoch': best_val_epoch,
        'best_val_acc': val_auc
    }

    if not save_str is None and save_dict['best_val_acc'] > save_threshold:
        save_threshold = save_dict['best_val_acc']
        torch.save({
            'encoder_state_dict': save_dict['encoder_state_dict'],
            'gnn_state_dict': model.state_dict()
        }, '%s_best_model.ckpt' % (save_str))
        print('update model with val acc %.4f to %s' % (save_threshold, '%s_best_model.ckpt' % (save_str)))

    return val_1, val_acc, val_ap, val_auc, test_1, test_acc, test_ap, test_auc


def evaluete(model, loaders, num_classes, lr_f=0.005, weight_decay_f=1e-5, max_epoch_f=100, device='cuda',
             linear_prob=True, mute=False, save_str=None):
    global save_threshold
    model.eval()
    if linear_prob:
        if len(loaders[0]) > 1:
            x_all = {"train": [], "val": [], "test": []}
            y_all = {"train": [], "val": [], "test": []}

            with torch.no_grad():
                for key, loader in zip(["train", "val", "test"], loaders):
                    for subgraph in loader:
                        subgraph = subgraph[0].to(device)
                        feat = subgraph.ndata["feat"]
                        x = model.embed(subgraph, feat)
                        x_all[key].append(x)
                        y_all[key].append(subgraph.ndata["label"])
            in_dim = x_all["train"][0].shape[1]
            encoder = LogisticRegression(in_dim, num_classes)
            num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
            if not mute:
                print(f"num parameters for finetuning: {sum(num_finetune_params)}")
                # torch.save(x.cpu(), "feat.pt")

            for key in ["train", "val", "test"]:
                y_key = torch.cat(y_all[key])
                plab = (y_key[:, 1] == 1).sum() / len(y_key)
                print('In %s, positive labels take up %.4f' % (key, plab))

            for key in ["train", "val", "test"]:
                y_key = torch.cat(y_all[key])
                plab = (y_key[:, 1] == 1).sum() / len(y_key)
                print('In %s, positive labels take up %.4f' % (key, plab))

            encoder.to(device)
            optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
            final_acc, estp_acc, save_dict = mutli_graph_linear_evaluation(encoder, x_all, y_all, optimizer_f,
                                                                           max_epoch_f, device,
                                                                           mute)

            if not save_str is None and save_dict['best_val_acc'] > save_threshold:
                save_threshold = save_dict['best_val_acc']
                torch.save({
                    'encoder_state_dict': save_dict['encoder_state_dict'],
                    'gnn_state_dict': model.state_dict()
                }, '%s_best_model.ckpt' % (save_str))

            return final_acc, estp_acc
        else:
            x_all = {"train": None, "val": None, "test": None}
            y_all = {"train": None, "val": None, "test": None}

            with torch.no_grad():
                for key, loader in zip(["train", "val", "test"], loaders):
                    for subgraph in loader:
                        subgraph = subgraph[0].to(device)
                        feat = subgraph.ndata["feat"]
                        x = model.embed(subgraph, feat)
                        mask = subgraph.ndata[f"{key}_mask"]
                        x_all[key] = x[mask]
                        y_all[key] = subgraph.ndata["label"][mask]
            in_dim = x_all["train"].shape[1]

            for key in ["train", "val", "test"]:
                y_key = torch.cat(y_all[key])
                plab = (y_key[:, 1] == 1).sum() / len(y_key)
                print('In %s, positive labels take up %.4f' % (key, plab))

            for key in ["train", "val", "test"]:
                y_key = torch.cat(y_all[key])
                plab = (y_key[:, 1] == 1).sum() / len(y_key)
                print('In %s, positive labels take up %.4f' % (key, plab))

            encoder = LogisticRegression(in_dim, num_classes)
            encoder = encoder.to(device)
            optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)

            x = torch.cat(list(x_all.values()))
            y = torch.cat(list(y_all.values()))
            num_train, num_val, num_test = [x.shape[0] for x in x_all.values()]
            num_nodes = num_train + num_val + num_test
            train_mask = torch.arange(num_train, device=device)
            val_mask = torch.arange(num_train, num_train + num_val, device=device)
            test_mask = torch.arange(num_train + num_val, num_nodes, device=device)

            final_acc, estp_acc, save_dict = linear_probing_for_inductive_node_classiifcation(encoder, x, y,
                                                                                              (train_mask, val_mask,
                                                                                               test_mask),
                                                                                              optimizer_f, max_epoch_f,
                                                                                              device,
                                                                                              mute)

            if not save_str is None and save_dict['best_val_acc'] > save_threshold:
                save_threshold = save_dict['best_val_acc']
                torch.save({
                    'encoder_state_dict': save_dict['encoder_state_dict'],
                    'gnn_state_dict': model.state_dict()
                }, '%s_best_model.ckpt' % (save_str))

            return final_acc, estp_acc
    else:
        raise NotImplementedError


def mutli_graph_linear_evaluation(model, feat, labels, optimizer, max_epoch, device, mute=False):
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = lambda x, y: sigmoid_focal_loss(x, y, reduction='mean')

    best_val_acc = 0
    best_val_epoch = 0
    best_val_test_acc = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        for x, y in zip(feat["train"], labels["train"]):
            out = model(None, x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

        with torch.no_grad():
            model.eval()
            val_out = []
            test_out = []
            for x, y in zip(feat["val"], labels["val"]):
                val_pred = model(None, x)
                val_out.append(val_pred)
            val_out = torch.cat(val_out, dim=0).cpu().numpy()
            val_label = torch.cat(labels["val"], dim=0).cpu().numpy()
            val_out = np.where(val_out >= 0, 1, 0)

            val_acc = f1_score(val_label, val_out, average="micro")

            for x, y in zip(feat["test"], labels["test"]):
                test_pred = model(None, x)  #
                test_out.append(test_pred)
            test_out = torch.cat(test_out, dim=0).cpu().numpy()
            test_label = torch.cat(labels["test"], dim=0).cpu().numpy()
            test_out = np.where(test_out >= 0, 1, 0)

            test_acc = f1_score(test_label, test_out, average="micro")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_val_test_acc = test_acc
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc}, test_acc:{test_acc: .4f}")

    # save
    save_dict = {
        'encoder_state_dict': best_model.state_dict(),
        # 'best_val_epoch': best_val_epoch,
        'best_val_acc': best_val_acc
    }

    if mute:
        print(
            f"# IGNORE: --- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f},  Final-TestAcc: {test_acc:.4f}--- ")
    else:
        print(
            f"--- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f}, Final-TestAcc: {test_acc:.4f} --- ")

    return test_acc, best_val_test_acc, save_dict


def node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device,
                                   linear_prob=True, mute=False):
    model.eval()
    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc, save_dict = linear_probing_for_transductive_node_classiifcation(encoder, graph, x, optimizer_f,
                                                                                         max_epoch_f, device, mute)
    return final_acc, estp_acc, save_dict


def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizer, max_epoch, device, mute=False):
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = lambda x, y: sigmoid_focal_loss(x, y, reduction='mean')

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x)

            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])

    # save
    save_dict = {
        'encoder_state_dict': best_model.state_dict(),
        # 'best_val_epoch': best_val_epoch,
        'best_val_acc': best_val_acc
    }

    if mute:
        print(
            f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(
            f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc, save_dict


def linear_probing_for_inductive_node_classiifcation(model, x, labels, mask, optimizer, max_epoch, device, mute=False):
    # if len(labels.shape) > 1:
    #     criterion = torch.nn.BCEWithLogitsLoss()
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()
    criterion = lambda x, y: sigmoid_focal_loss(x, y, reduction='mean')
    train_mask, val_mask, test_mask = mask

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

        best_val_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(None, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(None, x)
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(None, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])

    # save
    save_dict = {
        'encoder_state_dict': best_model.state_dict(),
        # 'best_val_epoch': best_val_epoch,
        'best_val_acc': best_val_acc
    }

    if mute:
        print(
            f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} ")
    else:
        print(
            f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}")

    return test_acc, estp_test_acc, save_dict


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits