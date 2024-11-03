from torch_geometric.nn import GCNConv, APPNP
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
import torch
from total_communicability import total_com
from Power_method import power
from katz_centrality import katz

class total_communicability_pool(torch.nn.Module):
    def __init__(self, args, in_channels, ratio):
        super(total_communicability_pool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.linearlayer = torch.nn.Linear(in_channels,1)
        self.score_layer = total_com(args.K, 1)

        self.non_linearity = torch.tanh
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        score = self.linearlayer(x)
        score = self.score_layer(score,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
    

class katz_pool(torch.nn.Module):
    def __init__(self, args, in_channels, ratio):
        super(katz_pool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.linearlayer = torch.nn.Linear(in_channels,1)
        self.score_layer = katz(args.K, args.alpha)

        self.non_linearity = torch.tanh
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        score = self.linearlayer(x)
        score = self.score_layer(score,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
    

class power_pool(torch.nn.Module):
    def __init__(self, args, in_channels, ratio):
        super(power_pool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.linearlayer = torch.nn.Linear(in_channels,1)
        self.score_layer = power(args.K, 1)

        self.non_linearity = torch.tanh
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        score = self.linearlayer(x)
        score = self.score_layer(score,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
    
class gcn_pool(torch.nn.Module):
    def __init__(self, in_channels, ratio):
        super(gcn_pool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        self.score_layer = GCNConv(in_channels, 1)

        self.non_linearity = torch.tanh
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
    

class appnp_pool(torch.nn.Module):
    def __init__(self, args, in_channels, ratio):
        super(appnp_pool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        self.linearlayer = torch.nn.Linear(in_channels,1)
        self.score_layer = APPNP(args.K, args.alpha)

        self.non_linearity = torch.tanh
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        score = self.linearlayer(x)
        score = self.score_layer(score,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm