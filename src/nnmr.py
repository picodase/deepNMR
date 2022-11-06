# Contruct a two-layer GNN model
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

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

def train(graph, node_features, node_labels, n_epochs=10):
    """
    Train a SAGE neural network
    """
    model = SAGE(in_feats=node_features.shape[1], hid_feats=100, out_feats=node_labels.shape[0])
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        model.train()
        
        # forward propagation by using all nodes
        logits = model(graph, node_features)
        
        # compute loss
        loss = F.cross_entropy(logits[graph.ndata["train_mask"][0]], node_labels[graph.ndata["train_mask"][0]])
        
        # compute validation accuracy
        acc = evaluate(model, graph, node_features, node_labels, graph.ndata["valid_mask"])
        
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())

        # Save model if necessary.  Omitted in this example

