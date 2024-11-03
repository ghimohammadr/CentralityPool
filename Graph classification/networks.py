import torch 
from torch import nn
from torch_geometric.nn import GCNConv, APPNP, GINConv, GATConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from layers import total_communicability_pool, gcn_pool, appnp_pool, power_pool, katz_pool
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, enhance=False):
        super(MLP, self).__init__()

        self.enhance = enhance

        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        if enhance:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        if self.enhance:
            x = self.bn1(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)

        x = self.fc2(x)
        if self.enhance:
            x = self.bn2(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)

        x = self.fc3(x)

        return x

class hireNet(torch.nn.Module):
    def __init__(self,args):
        super(hireNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio



        


        if args.poolmodel == 'gcnpool':
            convpool = gcn_pool
        elif args.poolmodel == 'tcommpool':
            convpool = total_communicability_pool
        elif args.poolmodel == 'appnppool':
            convpool = appnp_pool
        elif args.poolmodel == 'powerpool':
            convpool = power_pool
        elif args.poolmodel == 'katzpool':
            convpool = katz_pool


        if args.convmodel == 'gcn':
            # convlayer = GCNConv
            self.conv1 = GCNConv(self.num_features, self.nhid)
            self.pool1 = convpool(self.args, self.nhid, ratio=self.pooling_ratio)
            self.conv2 = GCNConv(self.nhid, self.nhid)
            self.pool2 = convpool(self.args, self.nhid, ratio=self.pooling_ratio)
            self.conv3 = GCNConv(self.nhid, self.nhid)
            self.pool3 = convpool(self.args, self.nhid, ratio=self.pooling_ratio)
        elif args.convmodel == 'gat':
            self.conv1 = GATConv(self.num_features, self.nhid, heads=8, dropout=0.6)
            self.pool1 = convpool(self.args, self.nhid * 8, ratio=self.pooling_ratio)
            self.conv2 = GATConv(self.nhid * 8, self.nhid, heads=8, dropout=0.6)
            self.pool2 = convpool(self.args, self.nhid * 8, ratio=self.pooling_ratio)
            self.conv3 = GATConv(self.nhid * 8, self.nhid, heads=1, dropout=0.6)
            self.pool3 = convpool(self.args, self.nhid, ratio=self.pooling_ratio)
        elif args.convmodel == "gin":
            self.conv1 = GINConv(torch.nn.Linear(self.num_features, self.nhid))
            self.pool1 = convpool(self.args, self.nhid, ratio=self.pooling_ratio)
            self.conv2 = GINConv(torch.nn.Linear(self.nhid, self.nhid))
            self.pool2 = convpool(self.args, self.nhid, ratio=self.pooling_ratio)
            self.conv3 = GINConv(torch.nn.Linear(self.nhid, self.nhid))
            self.pool3 = convpool(self.args, self.nhid, ratio=self.pooling_ratio)


        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()



    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)


        # x = x1 + x2 + x3 + x4
        x = x1 + x2 + x3 

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

    


class globNet(torch.nn.Module):
    def __init__(self,args):
        super(globNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.heads = 8


        if args.poolmodel == 'gcnpool':
            convpool = gcn_pool
        elif args.poolmodel == 'tcommpool':
            convpool = total_communicability_pool
        elif args.poolmodel == 'appnppool':
            convpool = appnp_pool
        elif args.poolmodel == 'powerpool':
            convpool = power_pool
        elif args.poolmodel == 'katzpool':
            convpool = katz_pool


        if args.convmodel == "gcn":
            convlayer = GCNConv
            self.conv1 = convlayer(self.num_features, self.nhid)
            self.conv2 = convlayer(self.nhid, self.nhid)
            self.conv3 = convlayer(self.nhid, self.nhid)
        elif args.convmodel == "gat":
            self.conv1 = GATConv(self.num_features, self.nhid, heads=8, dropout=0.6)
            self.conv2 = GATConv(self.nhid * 8, self.nhid, heads=8, dropout=0.6)
            self.conv3 = GATConv(self.nhid * 8, self.nhid, heads=1, concat=False, dropout=0.6)
        


        # self.conv1 = convlayer(self.num_features, self.nhid)
        # self.conv2 = convlayer(self.nhid, self.nhid)
        # self.conv3 = convlayer(self.nhid, self.nhid)


        self.pool1 = convpool(self.args, 3*self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid*2*3, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()



    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))

        # Concatenate x1, x2, x3 along the feature dimension
        concatenated_x = torch.cat([x1, x2, x3], dim=1)

        x, edge_index, _, batch, _ = self.pool1(concatenated_x, edge_index, None, batch)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x





