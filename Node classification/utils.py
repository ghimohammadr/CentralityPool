from torch_geometric.datasets import Planetoid
import torch
from torch_geometric.utils import to_dense_adj, coalesce, remove_self_loops
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Amazon, CitationFull, LastFMAsia, WikipediaNetwork, WebKB, AttributedGraphDataset, Actor, Coauthor
import random
import torch_geometric.transforms as T
# from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_undirected
import numpy as np
from torch_geometric.transforms import NormalizeFeatures

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the first dataset
def getData(dataset):
    if dataset  == 'BlogCatalog':
        dataset = AttributedGraphDataset(root='AttributedGraphDataset/dataset', name=dataset)
    elif dataset  == 'Actor':
        dataset = Actor(root='Actor/dataset')
    elif dataset  == 'Chameleon':
        dataset = WikipediaNetwork(root='WikipediaNetwork/dataset', name=dataset)
    elif dataset  == 'Squirrel':
        dataset = WikipediaNetwork(root='WikipediaNetwork/dataset', name=dataset)
    elif dataset == 'texas':
        dataset = WebKB(root='./dataset', name=dataset)
    elif dataset == 'wisconsin':
        dataset = WebKB(root='./dataset', name=dataset)
    elif dataset == 'cornell':
        dataset = WebKB(root='./dataset', name=dataset)
    elif dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=dataset)
    elif dataset == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=dataset)
    elif dataset == 'LastFMAsia':
        dataset = LastFMAsia(root='data/LastFMAsia')
    elif dataset == 'Corafull':
        dataset = CitationFull(root='data/CitationFull', name='Cora')
    elif dataset == 'Computers':
        dataset = Amazon(root='data/Amazon', name=dataset)
    elif dataset == 'photo':
        dataset = Amazon(root='data/Amazon', name=dataset)
    # elif dataset == "arxiv":
    #     dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    elif dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='./dataset', name=dataset, transform=NormalizeFeatures())
    data = dataset[0]
    
    return data, data.x.shape[1], len(set(np.array(data.y)))  




def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def splits(data, num_classes, exp, dataset):
    if exp == 'semisupervised':
        data = data
    if exp in ['random', 'fullsupervised']:
        indices = []
        for i in range(num_classes):
            index = torch.nonzero(torch.eq(data.y, i)).squeeze()
            # index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
        
        random.seed(42)
        random.shuffle(indices)

        if exp == 'fullsupervised':
            train_index = torch.cat([i[:int(0.6*len(i))] for i in indices], dim=0)
            val_index = torch.cat([i[int(0.6*len(i)):int(0.8*len(i))] for i in indices], dim=0)
            test_index = torch.cat([i[int(0.8*len(i)):] for i in indices], dim=0)
        elif exp == 'random':
            train_index = torch.cat([i[:int(0.025*len(i))] for i in indices], dim=0)
            val_index = torch.cat([i[int(0.025*len(i)):int(0.05*len(i))] for i in indices], dim=0)
            test_index = torch.cat([i[int(0.5*len(i)):] for i in indices], dim=0)

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    elif exp == 'arxiv':
        split_idx = dataset.get_idx_split()
        train_index = split_idx['train']
        val_index = split_idx['valid']
        test_index = split_idx['test']

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data


def load_data(dataset, exp):
    if dataset  == 'BlogCatalog':
        dataset = AttributedGraphDataset(root='AttributedGraphDataset/dataset', name=dataset)
    elif dataset  == 'Actor':
        dataset = Actor(root='Actor/dataset')
    elif dataset  == 'Chameleon':
        dataset = WikipediaNetwork(root='WikipediaNetwork/dataset', name=dataset)
    elif dataset  == 'Squirrel':
        dataset = WikipediaNetwork(root='WikipediaNetwork/dataset', name=dataset)
    elif dataset == 'texas':
        dataset = WebKB(root='./dataset', name=dataset)
    elif dataset == 'wisconsin':
        dataset = WebKB(root='./dataset', name=dataset)
    elif dataset == 'cornell':
        dataset = WebKB(root='./dataset', name=dataset)
    elif dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=dataset)
    elif dataset == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=dataset)
    elif dataset == 'LastFMAsia':
        dataset = LastFMAsia(root='data/LastFMAsia')
    elif dataset == 'Corafull':
        dataset = CitationFull(root='data/CitationFull', name='Cora')
    elif dataset == 'Computers':
        dataset = Amazon(root='data/Amazon', name=dataset)
    elif dataset == 'photo':
        dataset = Amazon(root='data/Amazon', name=dataset)
    # elif dataset == "arxiv":
    #     dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    elif dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='./dataset', name=dataset, transform=NormalizeFeatures())
    data = dataset[0]
    n_classes = len(set(np.array(dataset[0].y)))
    # data = splits(data, n_classes, exp, dataset)
    train_mask = data.train_mask
    val_mask = data.val_mask
    labels = data.y
    features = data.x



    # # x ,y = data.x, data.y[:, 0]
    # # scaler = StandardScaler()
    # # x = torch.from_numpy(scaler.fit_transform(x.numpy()))
    # # data.edge_index = to_undirected(data.edge_index)
    # # data.edge_index = coalesce(data.edge_index)
    # # data.edge_index = remove_self_loops(data.edge_index)[0]
    # # data = Data(x=x, edge_index=data.edge_index, y=y)

    # # # n_classes = dataset.num_classes
    # # # data = splits(data, n_classes, exp, dataset)
    # # train_mask = data.train_mask
    # # val_mask = data.val_mask
    # # labels = data.y
    # # features = data.x


    return data, features, train_mask, val_mask, labels
