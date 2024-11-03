import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch_geometric import transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
from total_communicability import total_com
from Power_method import power
from katz_centrality import katz
import random
import optuna
from optuna.samplers import GridSampler
import argparse
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_dense_adj, coalesce, remove_self_loops
import builtins
seed = 777
random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def splits(data, num_classes, exp):
    if exp != 'fixed':
        indices = []
        for i in range(num_classes):
            # index = torch.nonzero(torch.eq(data.y, i)).squeeze()
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            print(index)
            indices.append(index)
        
        random.seed(seed)
        random.shuffle(indices)
        if exp == 'fullsupervised':
            train_index = torch.cat([i[:int(0.6*len(i))] for i in indices], dim=0)
            val_index = torch.cat([i[int(0.6*len(i)):int(0.8*len(i))] for i in indices], dim=0)
            test_index = torch.cat([i[int(0.8*len(i)):] for i in indices], dim=0)
        elif exp == 'semisupervised':
            train_index = torch.cat([i[:int(0.025*len(i))] for i in indices], dim=0)
            val_index = torch.cat([i[int(0.025*len(i)):int(0.05*len(i))] for i in indices], dim=0)
            test_index = torch.cat([i[int(0.05*len(i)):] for i in indices], dim=0)
        elif exp == 'piesplit':
            train_index = torch.cat([i[:int(0.48*len(i))] for i in indices], dim=0)
            val_index = torch.cat([i[int(0.48*len(i)):int(0.8*len(i))] for i in indices], dim=0)
            test_index = torch.cat([i[int(0.8*len(i)):] for i in indices], dim=0)

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data



dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0]


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

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, training=self.training)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate(runs, NETmodel, K, alpha, hidden, epochs, early_stopping, lr, weight_decay, normalize_features, dataset, seed):
    results = []
    for run in range(runs):
        model = Net(dataset.num_node_features, hidden, dataset.num_classes, K, alpha, NETmodel).to(device)
        # model.reset_parameters()
        # data = splits(data, n_classes, 'fullsupervised')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        for epoch in range(1, epochs + 1):
            train(model, data, optimizer)
        
        _, _, test_acc = test(model, data)

        results.append(test_acc)
        print("Test accuracy and std: ", np.mean(results), np.std(results))

    return np.mean(results)

def objective(trial):
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    alpha = trial.suggest_float('alpha', 0.5, 1, log=False)
    epochs = trial.suggest_int('epochs', 80, 500)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    hidden = trial.suggest_int('hidden', 32, 512)
    K = trial.suggest_int('K', 10, 50)
    early_stopping = trial.suggest_int('early_stopping', 10, 50)


    


    return train_and_evaluate(
        runs=10,
        NETmodel='katzcentrality',
        K=K,
        alpha=alpha,
        hidden=hidden,
        epochs=epochs,
        early_stopping=early_stopping,
        lr=lr,
        weight_decay=weight_decay,
        normalize_features='True',
        dataset=dataset,
        seed=777
    )

# Optimization with GridSampler
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    search_space = {
        'lr': [1e-4, 1e-2],
        'alpha': [0.8],
        'epochs': [100, 300],
        'weight_decay': [1e-4, 1e-3],
        'hidden': [64, 128],
        'K': [10],
        'early_stopping': [10, 20]

    }

    study = optuna.create_study(direction='maximize', sampler=GridSampler(search_space))
    study.optimize(objective, n_trials=3*4*3*2*3*3*2)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
