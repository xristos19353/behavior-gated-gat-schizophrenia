"""Graph neural network models and training utilities.

Two architectures are provided:

* :class:`GAT` - graph attention network with an optional behaviour-conditioned
  attention-pooling gate. The gate concatenates the graph-level behavioural
  features to each node embedding and learns a per-node pooling weight, which
  also serves as a node-importance explanation.
* :class:`GCN` - graph convolutional baseline that concatenates the graph-level
  features after mean pooling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, LayerNorm, global_mean_pool
from torch_geometric.utils import softmax


class EarlyStopping:
    """Stop training when the monitored score stops improving."""

    def __init__(self, patience=25, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def step(self, current_score):
        if self.best_score is None or current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class GAT(torch.nn.Module):
    """Graph attention network with optional behaviour-gated attention pooling."""

    def __init__(self, in_channels, hidden_channels, dropout, graph_feat_dim,
                 num_layers=2, heads=1, use_layernorm=False, concat=True,
                 pool="attn", use_edge_attr=False):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.use_layernorm = use_layernorm
        self.dropout = dropout
        self.concat = concat
        self.heads = heads
        self.use_edge_attr = use_edge_attr

        edge_dim = 1 if use_edge_attr else None
        out_dim = hidden_channels * heads if concat else hidden_channels

        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads,
                                    dropout=dropout, concat=concat, edge_dim=edge_dim))
        if use_layernorm:
            self.norms.append(LayerNorm(out_dim))

        for _ in range(num_layers - 1):
            self.layers.append(GATConv(out_dim, hidden_channels, heads=heads,
                                        dropout=dropout, concat=concat, edge_dim=edge_dim))
            if use_layernorm:
                self.norms.append(LayerNorm(out_dim))

        self.pool = pool
        if pool == "attn":
            self.gate_nn = torch.nn.Sequential(
                torch.nn.Linear(out_dim + graph_feat_dim, out_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(out_dim, 1),
            )

        self.fc = torch.nn.Linear(out_dim, 1)

    def forward(self, x, edge_index, graph_feat, batch=None, edge_attr=None,
                return_attn=False, return_pool_weights=False):
        attn_weights_list = []

        for i, conv in enumerate(self.layers):
            ea = edge_attr if self.use_edge_attr else None
            if return_attn:
                x, (edge_index_out, attn_weights) = conv(
                    x, edge_index, edge_attr=ea, return_attention_weights=True
                )
                attn_weights_list.append((edge_index_out, attn_weights))
            else:
                x = conv(x, edge_index, edge_attr=ea)

            if self.use_layernorm:
                x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pool_weights = None
        if self.pool == "attn":
            c_per_node = graph_feat[batch]
            gate_in = torch.cat([x, c_per_node], dim=1)
            gate = self.gate_nn(gate_in).view(-1)

            alpha = softmax(gate, batch)
            if return_pool_weights:
                pool_weights = alpha

            x = x * alpha.unsqueeze(-1)

            batch_size = int(batch.max().item()) + 1
            x_pooled = torch.zeros(batch_size, x.size(1), device=x.device)
            x_pooled.index_add_(0, batch, x)
            x = x_pooled
        else:
            x = global_mean_pool(x, batch)

        out = self.fc(x)

        if return_attn and return_pool_weights:
            return out, attn_weights_list, pool_weights
        if return_attn:
            return out, attn_weights_list
        if return_pool_weights:
            return out, pool_weights
        return out


class GCN(torch.nn.Module):
    """Graph convolutional baseline with concatenated graph-level features."""

    def __init__(self, in_channels, hidden_channels, dropout, graph_feat_dim, num_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels, normalize=False))

        self.fc = torch.nn.Linear(hidden_channels + graph_feat_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, graph_feat, batch=None, edge_weight=None):
        for conv in self.layers:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = torch.cat([x, graph_feat], dim=1)
        return self.fc(x)


def build_model(model_type, in_channels, params, graph_feat_dim, use_edge_attr=False):
    """Construct a GAT or GCN from a hyperparameter dictionary."""
    if model_type == "GCN":
        return GCN(
            in_channels=in_channels,
            hidden_channels=params["hidden_dim"],
            dropout=params["dropout"],
            graph_feat_dim=graph_feat_dim,
            num_layers=params["num_layers"],
        )
    return GAT(
        in_channels=in_channels,
        hidden_channels=params["hidden_dim"],
        dropout=params["dropout"],
        graph_feat_dim=graph_feat_dim,
        num_layers=params["num_layers"],
        heads=params["heads"],
        use_edge_attr=use_edge_attr,
    )
