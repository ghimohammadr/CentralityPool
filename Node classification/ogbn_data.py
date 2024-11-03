import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
from total_communicability import total_com
from Power_method import power
from katz_centrality import katz
from torch_geometric.nn import APPNP


# Custom Model Class
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K, alpha, NETmodel):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        if NETmodel == 'katzcentrality':
            self.prop1 = katz(K, alpha)
        elif NETmodel == 'appnp':
            self.prop1 = APPNP(K, alpha)
        elif NETmodel == 'totalcentrality':
            self.prop1 = total_com(K, alpha)
        elif NETmodel == 'powercentrality':
            self.prop1 = power(K, alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, adj_t):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, training=self.training)
        x = self.prop1(x, adj_t)
        return F.log_softmax(x, dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


import itertools

def main():
    parser = argparse.ArgumentParser(description='OGBN-arxiv (GNN)')    #ogbn-products
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--NETmodel', type=str, default='katzcentrality')
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--K', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor()) #ogbn-products
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    evaluator = Evaluator(name='ogbn-arxiv') #ogbn-products

    # Define parameter grids for hidden_channels, lr, and epochs
    hidden_channels_grid = [256, 512, 1024]
    lr_grid = [0.001, 0.01]
    epochs_grid = [500, 1000, 2000]
    # hidden_channels_grid = [512]
    # lr_grid = [0.001, 0.01, 0.05]
    # epochs_grid = [200, 500, 1000]

    # Generate all combinations of these values
    param_combinations = list(itertools.product(hidden_channels_grid, lr_grid, epochs_grid))

    # Loop over all parameter combinations
    for hidden_channels, lr, epochs in param_combinations:
        print(f"Running for hidden_channels={hidden_channels}, lr={lr}, epochs={epochs}")

        if args.use_sage:
            model = Net(data.num_features, hidden_channels, dataset.num_classes, args.K, args.alpha, args.NETmodel).to(device)
        else:
            # model = GCN(data.num_features, hidden_channels, dataset.num_classes, args.num_layers, args.dropout).to(device)
            model = Net(data.num_features, hidden_channels, dataset.num_classes, args.K, args.alpha, args.NETmodel).to(device)

        logger = Logger(args.runs, args)

        for run in range(args.runs):
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            for epoch in range(1, 1 + epochs):
                loss = train(model, data, train_idx, optimizer)
                result = test(model, data, split_idx, evaluator)
                logger.add_result(run, result)

            logger.print_statistics(run)
        logger.print_statistics()



if __name__ == "__main__":
    main()