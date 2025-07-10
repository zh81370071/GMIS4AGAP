import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import torch_geometric.nn as gnn
from copy import deepcopy
import random
import os
import numpy as np
import pickle
import networkx as nx
import torch_scatter
import os
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
def clones(layer, n):
    return nn.ModuleList([deepcopy(layer) for _ in range(n)])


# GNN for node embeddings
class NodeEmbNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, act_fn='silu', norm=gnn.BatchNorm):
        super(NodeEmbNet, self).__init__()
        self.lins = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(2)])
        self.act_fn = getattr(F, act_fn)
        self.norm = norm(out_dim)

    def forward(self, h, e, edge_index):
        h1 = self.lins[0](h)  # [num_nodes, out_dim]
        h2 = self.lins[1](h)  # [num_nodes, out_dim]

        weighted_messages = torch.sigmoid(e) * h2[edge_index[1]]  # [num_edges, out_dim]

        agg = torch_scatter.scatter_add(weighted_messages, edge_index[0], dim=0, dim_size=h.size(0))  # [num_nodes, out_dim]

        x = h1 + agg

        x = self.act_fn(self.norm(x))

        return h + x


class EdgeEmbNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, act_fn='silu', norm=gnn.BatchNorm):
        super(EdgeEmbNet, self).__init__()
        self.lins = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(3)])
        self.act_fn = getattr(F, act_fn)
        self.norm = norm(out_dim)

    def forward(self, h, e, edge_index):
        # center /neighborhood representation
        h1, h2 = self.lins[0](h), self.lins[1](h)
        # edge representation
        x = self.lins[-1](e) + h1[edge_index[0]] + h2[edge_index[1]]
        # norm and activation
        residual = self.act_fn(self.norm(x))
        # residual connection
        return e + residual


class PredictionHead(nn.Module):
    def __init__(self, dim, pooling, act_fn):
        super(PredictionHead, self).__init__()
        # attention matrix
        self.W = nn.Parameter(torch.empty(dim, dim))
        # initialize
        nn.init.kaiming_normal_(self.W)
        # pooling
        self.pooling = pooling
        # active function
        self.act_fn = act_fn

    def forward(self, h):
        # h: n x d -> (d,1 )
        x = self.pooling(h)
        # (n,d) @ (d,d) @ (d,1) -> (n,1)
        return self.act_fn(h @ self.W @ x)


class Net(torch.nn.Module):
    def __init__(self, embeddings, node_emb_nets, edge_emb_nets, pred):
        super(Net, self).__init__()
        # node / edge initial embedding
        self.embeddings = embeddings
        # node embedding nets
        self.node_emb_nets = node_emb_nets
        # edge embedding nets
        self.edge_emb_nets = edge_emb_nets
        # prediction head
        self.pred = pred

    def forward(self, x, edge_index, edge_attr):
        # embedding
        h, e = self.embeddings[0](x), self.embeddings[-1](edge_attr)
        # deep representation
        for node_emb, edge_emb in zip(self.node_emb_nets, self.edge_emb_nets):
            # node/edge aggregation
            h, e = node_emb(h, e, edge_index), edge_emb(h, e, edge_index)
        # yield the prediction representation
        return self.pred(h)


class AvgPooling(nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, x):
        # (n,d) -> (d,1)
        return torch.t(torch.mean(x, dim=0, keepdim=True))


class Policy(object):
    def __init__(self, in_dim, out_dim, act_fn, agg_fn, norm, depth,
                 initial_lr, step_size, gamma, tradeoff, pooling):
        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model
        self.model = Net(nn.ModuleList([nn.Linear(in_dim, out_dim), nn.Linear(3, out_dim)]),
                         clones(NodeEmbNet(out_dim, out_dim, act_fn, norm), depth - 1),
                         clones(EdgeEmbNet(out_dim, out_dim, act_fn, norm), depth - 1),
                         PredictionHead(out_dim, pooling, F.sigmoid)).to(self.device)
        # optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=initial_lr)
        # scheduler for learning rate
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def train_epoch(self, samples, batch_size):
        # total loss
        total_loss = 0
        # randomly shuffle the samples
        random.shuffle(samples)
        # sample batch_size
        for idx in range(len(samples) // batch_size):
            # instances
            instances = samples[idx * batch_size:(idx + 1) * batch_size]
            # accumulative loss
            loss = 0
            # iterative process each graph (instance),
            for G in instances:
                G.to(self.device)
                probs = self.model(G.x, G.edge_index, G.edge_attr)
                loss += probs.T @ G.Q @ probs

            # zero gradient
            self.optimizer.zero_grad()
            # backward
            loss.backward()
            # step
            self.optimizer.step()
            # accumulate the loss
            total_loss += loss.item()
        # return the total loss
        return total_loss


    def train(self, samples, epochs, batch_size, path):
        losses = []  # 用于保存每个epoch的loss

        for epoch in range(epochs):
            loss = self.train_epoch(samples, batch_size)
            self.scheduler.step(epoch)
            if not os.path.exists(path):
                os.mkdir(path)

            torch.save(self.model.state_dict(), os.path.join(path, f"{epoch}.pt"))
            print(f"epoch {epoch}\t loss {loss:.3f}")
            losses.append(loss)

            # 训练结束后绘制loss曲线
        plt.figure()
        plt.plot(range(epochs), losses, label='train loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve for GNP4MIS')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(path, 'train_loss_GNP.png'))  # 保存图像
        plt.show()