# Contruct a two-layer GNN model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn


### CONSTANTS

REPORT = 50
DROPOUT = 0#0.15

### DEFINITIONS

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean', feat_drop=DROPOUT)
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean', feat_drop=DROPOUT)

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train(g, node_features, node_labels, n_epochs=10):
    """
    Train a SAGE neural network
    """
    model = SAGE(in_feats=node_features.shape[1], hid_feats=100, out_feats=node_labels.shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_err = 0
    best_test_err = 0

    train_mask = g.ndata['train_mask'].long()
    valid_mask = g.ndata['valid_mask'].long()
    test_mask = g.ndata['test_mask'].long()

    for e in range(n_epochs):
        # Forward
        logits = model(g, node_features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.mse_loss(logits[train_mask], node_labels[train_mask])

        # Compute accuracy on training/validation/test
        #train_err = (pred[train_mask] - node_labels[train_mask]).float().mean().square()
        #val_err = (pred[valid_mask] - node_labels[valid_mask]).float().mean().square()
        #test_err = (pred[test_mask] - node_labels[test_mask]).float().mean().square()

        val_err = F.mse_loss(pred[valid_mask], node_labels[valid_mask])
        test_err = F.mse_loss(pred[test_mask], node_labels[test_mask])

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_err > val_err:
            best_val_err = val_err
            best_test_err = test_err

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % REPORT == 0:
            print('In epoch {}, loss: {:.1e}, val err: {:.1e} (best {:.1e}), test err: {:.1e} (best {:.1e})'.format(
                e, loss, val_err, best_val_err, test_err, best_test_err))

    return model