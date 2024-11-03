import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
import numpy as np
from networks import hireNet, globNet
import torch.nn.functional as F
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from ogb.graphproppred import GraphPropPredDataset
import argparse
import os
import time
import random
import optuna
from optuna.samplers import GridSampler
from torch.utils.data import random_split

# Define the training and evaluation function
def train_and_evaluate(runs, poolmodel, convmodel, netmodel, seed, batch_size, lr, weight_decay, nhid, K, pooling_ratio, alpha, dropout_ratio, dataset, epochs, patience):
    args = argparse.Namespace(
        runs=runs,
        poolmodel=poolmodel,
        convmodel=convmodel,
        netmodel=netmodel,
        seed=seed,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        nhid=nhid,
        K=K,
        pooling_ratio=pooling_ratio,
        alpha=alpha,
        dropout_ratio=dropout_ratio,
        dataset=dataset,
        epochs=epochs,
        patience=patience,
        device='cpu'
    )
    

    all_acc = []
    start = time.time()
    
    

    for run in range(runs):

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            args.device = 'cuda:0'
    
        class NormalizedDegree(object):
            def __init__(self, mean, std):
                self.mean = mean
                self.std = std

            def __call__(self, data):
                deg = degree(data.edge_index[0], dtype=torch.float)
                deg = (deg - self.mean) / self.std
                data.x = deg.view(-1, 1)
                return data
            
        # Load the dataset
        # dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True, cleaned=False)
        dataset = GraphPropPredDataset(name = 'ogbg-molhiv')
        print(dataset)
        # dataset.data.edge_attr = None
        # max_degree = 0
        # degs = []
        # for data in dataset:
        #     degs += [degree(data.edge_index[0], dtype=torch.long)]
        #     max_degree = max(max_degree, degs[-1].max().item())

        # if max_degree < 1000:
        #     dataset.transform = T.OneHotDegree(max_degree)
        # else:
        #     deg = torch.cat(degs, dim=0).to(torch.float)
        #     mean, std = deg.mean().item(), deg.std().item()
        #     dataset.transform = NormalizedDegree(mean, std)

        args.num_classes = dataset.num_classes
        args.num_features = dataset.num_features

        num_training = int(len(dataset) * 0.8)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_training + num_val)
        training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])



        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        if args.netmodel == 'hire':
            model = hireNet(args).to(args.device)
        elif args.netmodel == 'glob':
            model = globNet(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.reset_parameters()

        def test(model, loader):
            model.eval()
            correct = 0.
            loss = 0.
            for data in loader:
                data = data.to(args.device)
                out = model(data)
                pred = out.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
                loss += F.nll_loss(out, data.y, reduction='sum').item()
                # loss += F.cross_entropy(out, data.y).item()
            return correct / len(loader.dataset), loss / len(loader.dataset)

        min_loss = 1e10
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()
            for i, data in enumerate(train_loader):
                data = data.to(args.device)
                out = model(data)
                loss = F.nll_loss(out, data.y)
                # loss = F.cross_entropy(out, data.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            val_acc, val_loss = test(model, val_loader)
            if val_loss < min_loss:
                torch.save(model.state_dict(), 'latest.pth')
                min_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter > args.patience:
                break


        model.load_state_dict(torch.load('latest.pth'))
        test_acc, test_loss = test(model, test_loader)
        print("Test accuracy:{}".format(test_acc))
        all_acc.append(test_acc)


    end = time.time()
    avg_acc = np.mean(all_acc)
    print('ave_acc: {:.4f}'.format(avg_acc), '+/- {:.4f}'.format(np.std(all_acc)))
    print('ave_time:', (end - start) / args.runs)
    print("args: \n", args)
    return avg_acc

# Define the objective function for Optuna
def objective(trial):
    # seeds = trial.suggest_int("seeds", 0, 1000),
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    alpha = trial.suggest_float('alpha', 0.1, 0.9)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    nhid = trial.suggest_int('nhid', 64, 2048)
    K = trial.suggest_int('K', 1, 100)
    ratio = trial.suggest_float('ratio', 0.1, 0.9)

    return train_and_evaluate(
        runs=5,
        poolmodel='katzpool', #tcommpool, appnppool, powerpool, katzpool
        convmodel='gcn',
        netmodel='hire',
        seed=777,
        # seeds=[777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800],
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        nhid=nhid,
        K=K,
        pooling_ratio=ratio,
        alpha=alpha,
        dropout_ratio=0.5,
        dataset='DD',
        epochs=1000,
        patience=50
    )

# Run the optimization with GridSampler
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_trials', type=int, default=2, help='number of trials for hyperparameter tuning')
    args = parser.parse_args()

    print(args)


    search_space = {
        'lr': [1e-5],
        'batch_size': [64],
        'weight_decay': [1e-4],
        'nhid': [1024],
        'ratio': [0.75],
        'alpha': [0.8],
        'K': [10]
    }

    study = optuna.create_study(direction='maximize', sampler=GridSampler(search_space))
    study.optimize(objective, n_trials=3*2*2*3*1)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))















# import torch
# from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
# from torch_geometric import utils
# import numpy as np
# from networks import hireNet, globNet
# import torch.nn.functional as F
# from torch_geometric.utils import degree
# import torch_geometric.transforms as T
# import argparse
# import os
# import time
# import random
# import optuna
# from optuna.samplers import GridSampler
# from torch.utils.data import random_split

# # Define the training and evaluation function
# def train_and_evaluate(runs, poolmodel, convmodel, netmodel, seeds, batch_size, lr, weight_decay, nhid, K, pooling_ratio, dropout_ratio, dataset, epochs, patience):
#     args = argparse.Namespace(
#         runs=runs,
#         poolmodel=poolmodel,
#         convmodel=convmodel,
#         netmodel=netmodel,
#         seeds=seeds,
#         batch_size=batch_size,
#         lr=lr,
#         weight_decay=weight_decay,
#         nhid=nhid,
#         K=K,
#         pooling_ratio=pooling_ratio,
#         dropout_ratio=dropout_ratio,
#         dataset=dataset,
#         epochs=epochs,
#         patience=patience,
#         device='cpu'
#     )
    

#     all_acc = []
#     start = time.time()
    
    

#     for seed in seeds:
#         print(seed)

#         # seed = int(seed)

#         torch.manual_seed(seed)
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

#         # Ensure deterministic behavior in cuDNN
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)
#             args.device = 'cuda:0'
    
#         class NormalizedDegree(object):
#             def __init__(self, mean, std):
#                 self.mean = mean
#                 self.std = std

#             def __call__(self, data):
#                 deg = degree(data.edge_index[0], dtype=torch.float)
#                 deg = (deg - self.mean) / self.std
#                 data.x = deg.view(-1, 1)
#                 return data
            
#         # Load the dataset
#         dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True, cleaned=False)
#         dataset.data.edge_attr = None
#         max_degree = 0
#         degs = []
#         for data in dataset:
#             degs += [degree(data.edge_index[0], dtype=torch.long)]
#             max_degree = max(max_degree, degs[-1].max().item())

#         if max_degree < 1000:
#             dataset.transform = T.OneHotDegree(max_degree)
#         else:
#             deg = torch.cat(degs, dim=0).to(torch.float)
#             mean, std = deg.mean().item(), deg.std().item()
#             dataset.transform = NormalizedDegree(mean, std)

#         args.num_classes = dataset.num_classes
#         args.num_features = dataset.num_features

#         num_training = int(len(dataset) * 0.8)
#         num_val = int(len(dataset) * 0.1)
#         num_test = len(dataset) - (num_training + num_val)
#         training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])



#         train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
#         val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
#         test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

#         if args.netmodel == 'hire':
#             model = hireNet(args).to(args.device)
#         elif args.netmodel == 'glob':
#             model = globNet(args).to(args.device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#         model.reset_parameters()

#         def test(model, loader):
#             model.eval()
#             correct = 0.
#             loss = 0.
#             for data in loader:
#                 data = data.to(args.device)
#                 out = model(data)
#                 pred = out.max(dim=1)[1]
#                 correct += pred.eq(data.y).sum().item()
#                 loss += F.nll_loss(out, data.y, reduction='sum').item()
#                 # loss += F.cross_entropy(out, data.y).item()
#             return correct / len(loader.dataset), loss / len(loader.dataset)

#         min_loss = 1e10
#         patience_counter = 0

#         for epoch in range(args.epochs):
#             model.train()
#             for i, data in enumerate(train_loader):
#                 data = data.to(args.device)
#                 out = model(data)
#                 loss = F.nll_loss(out, data.y)
#                 # loss = F.cross_entropy(out, data.y)
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#             val_acc, val_loss = test(model, val_loader)
#             if val_loss < min_loss:
#                 torch.save(model.state_dict(), 'latest.pth')
#                 min_loss = val_loss
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#             if patience_counter > args.patience:
#                 break


#         model.load_state_dict(torch.load('latest.pth'))
#         test_acc, test_loss = test(model, test_loader)
#         print("Test accuracy:{}".format(test_acc))
#         all_acc.append(test_acc)


#     end = time.time()
#     avg_acc = np.mean(all_acc)
#     print('ave_acc: {:.4f}'.format(avg_acc), '+/- {:.4f}'.format(np.std(all_acc)))
#     print('ave_time:', (end - start) / args.runs)
#     print("args: \n", args)
#     return avg_acc

# # Define the objective function for Optuna
# def objective(trial):
#     # seeds = trial.suggest_int("seeds", 0, 1000),
#     lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
#     batch_size = trial.suggest_int('batch_size', 32, 128)
#     weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
#     nhid = trial.suggest_int('nhid', 64, 2048)
#     # K = trial.suggest_int('K', 10, 100)
#     ratio = trial.suggest_float('ratio', 0.25, 0.75)

#     return train_and_evaluate(
#         runs=10,
#         poolmodel='katzpool', #tcommpool, appnppool, powerpool, katzpool
#         convmodel='gcn',
#         netmodel='hire',
#         seeds=[777, 778, 780, 781, 782, 784, 785, 791, 793, 799],
#         # seeds=[777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800],
#         batch_size=batch_size,
#         lr=lr,
#         weight_decay=weight_decay,
#         nhid=nhid,
#         K=10,
#         pooling_ratio=ratio,
#         dropout_ratio=0.5,
#         dataset='DD',
#         epochs=1000,
#         patience=50
#     )

# # Run the optimization with GridSampler
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--n_trials', type=int, default=2, help='number of trials for hyperparameter tuning')
#     args = parser.parse_args()

#     print(args)

#     search_space = {
#         'lr': [1e-5, 1e-4],
#         'batch_size': [64],
#         'weight_decay': [1e-3],
#         'nhid': [512, 1024],
#         'ratio': [0.75]
#     }

#     study = optuna.create_study(direction='maximize', sampler=GridSampler(search_space))
#     study.optimize(objective, n_trials=3*2*2*3*1)

#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: {}".format(trial.value))
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))
















# import torch
# from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
# from torch_geometric import utils
# import numpy as np
# from networks import hireNet, globNet
# import torch.nn.functional as F
# from torch_geometric.utils import degree
# import torch_geometric.transforms as T
# import numpy as np
# from sklearn.model_selection import KFold
# import argparse
# import os
# import time
# import random
# import optuna
# from optuna.samplers import GridSampler
# from torch.utils.data import random_split, Subset

# # Define the training and evaluation function
# def train_and_evaluate(fold, poolmodel, convmodel, netmodel, seed, batch_size, lr, weight_decay, nhid, K, pooling_ratio, dropout_ratio, dataset, epochs, patience):
#     args = argparse.Namespace(
#         fold=fold,
#         poolmodel=poolmodel,
#         convmodel=convmodel,
#         netmodel=netmodel,
#         seed=seed,
#         batch_size=batch_size,
#         lr=lr,
#         weight_decay=weight_decay,
#         nhid=nhid,
#         K=K,
#         pooling_ratio=pooling_ratio,
#         dropout_ratio=dropout_ratio,
#         dataset=dataset,
#         epochs=epochs,
#         patience=patience,
#         device='cpu'
#     )
    
#     # Set seeds for reproducibility
#     torch.manual_seed(args.seed)
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)

#     # Ensure deterministic behavior in cuDNN
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(args.seed)
#         args.device = 'cuda:0'


        
#     # Load the dataset
#     dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, cleaned=False)


#     args.num_classes = dataset.num_classes
#     args.num_features = dataset.num_features

#     # Perform 10-fold cross-validation
#     num_folds = 10
#     fold_size = len(dataset) // num_folds
#     indices = np.arange(len(dataset))

#     np.random.seed(args.seed)
#     np.random.shuffle(indices)

#     all_acc = []
#     start = time.time()


#     # Shuffle indices
#     indices = np.random.permutation(len(dataset))

#     kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)

#     for fold, (train_val_idx, test_idx) in enumerate(kf.split(indices)):
#         print(f"Fold {fold+1}/10")

#         # Split train_val into train and validation
#         val_size = len(train_val_idx) // 9
#         val_idx = train_val_idx[:val_size]
#         train_idx = train_val_idx[val_size:]
        

#         train_set = Subset(dataset, indices[train_idx])
#         val_set = Subset(dataset, indices[val_idx])
#         test_set = Subset(dataset, indices[test_idx])


#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
#         val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
#         test_loader = DataLoader(test_set, batch_size=1, shuffle=False)        


#         if args.netmodel == 'hire':
#             model = hireNet(args).to(args.device)
#         elif args.netmodel == 'glob':
#             model = globNet(args).to(args.device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#         model.reset_parameters()

#         def test(model, loader):
#             model.eval()
#             correct = 0.
#             loss = 0.
#             for data in loader:
#                 data = data.to(args.device)
#                 out = model(data)
#                 pred = out.max(dim=1)[1]
#                 correct += pred.eq(data.y).sum().item()
#                 loss += F.nll_loss(out, data.y, reduction='sum').item()
#             return correct / len(loader.dataset), loss / len(loader.dataset)

#         min_loss = 1e10
#         patience_counter = 0

#         for epoch in range(args.epochs):
#             model.train()
#             for i, data in enumerate(train_loader):
#                 data = data.to(args.device)
#                 out = model(data)
#                 loss = F.nll_loss(out, data.y)
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#             val_acc, val_loss = test(model, val_loader)
#             if val_loss < min_loss:
#                 torch.save(model.state_dict(), 'latest.pth')
#                 min_loss = val_loss
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#             if patience_counter > args.patience:
#                 break

#         model.load_state_dict(torch.load('latest.pth'))
#         test_acc, test_loss = test(model, test_loader)
#         print(f"Fold {fold+1} test accuracy: {test_acc}")
#         all_acc.append(test_acc)

#     end = time.time()
#     avg_acc = np.mean(all_acc)
#     print('Average accuracy: {:.4f}'.format(avg_acc), '+/- {:.4f}'.format(np.std(all_acc)))
#     print('Average time:', (end - start) / num_folds)
#     print("args: \n", args)
#     return avg_acc

# # Define the objective function for Optuna
# def objective(trial):
#     lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
#     batch_size = trial.suggest_int('batch_size', 32, 128)
#     weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
#     nhid = trial.suggest_int('nhid', 64, 2048)
#     ratio = trial.suggest_float('ratio', 0.25, 0.75)

#     return train_and_evaluate(
#         fold=10,
#         poolmodel='katzpool', #tcommpool, appnppool, powerpool, katzpool
#         convmodel='gcn',
#         netmodel='hire',
#         seed=788,
#         batch_size=batch_size,
#         lr=lr,
#         weight_decay=weight_decay,
#         nhid=nhid,
#         K=10,
#         pooling_ratio=ratio,
#         dropout_ratio=0.5,
#         dataset='DD',
#         epochs=1000,
#         patience=50
#     )

# # Run the optimization with GridSampler
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()

#     search_space = {
#         'lr': [1e-5, 1e-4, 1e-3],
#         'batch_size': [64, 128],
#         'weight_decay': [1e-4, 1e-3],
#         'nhid': [128, 512, 1024],
#         'ratio': [0.75]
#     }

#     study = optuna.create_study(direction='maximize', sampler=GridSampler(search_space))
#     study.optimize(objective, n_trials=3*2*2*3*1)

#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: {}".format(trial.value))
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))





























# import torch
# from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
# from torch_geometric import utils
# import numpy as np
# from networks import hireNet, globNet
# import torch.nn.functional as F
# from torch_geometric.utils import degree
# import torch_geometric.transforms as T
# import numpy as np
# from sklearn.model_selection import KFold
# import argparse
# import os
# import time
# import random
# import optuna
# from optuna.samplers import GridSampler
# from torch.utils.data import random_split, Subset
# from torch.nn.utils import clip_grad_norm_
# from torch.optim.lr_scheduler import LambdaLR
# from sklearn.model_selection import StratifiedKFold

# # Define the training and evaluation function
# def train_and_evaluate(fold, poolmodel, convmodel, netmodel, seed, batch_size, lr, weight_decay, nhid, K, pooling_ratio, dropout_ratio, dataset, epochs, patience):
#     args = argparse.Namespace(
#         fold=fold,
#         poolmodel=poolmodel,
#         convmodel=convmodel,
#         netmodel=netmodel,
#         seed=seed,
#         batch_size=batch_size,
#         lr=lr,
#         weight_decay=weight_decay,
#         nhid=nhid,
#         K=K,
#         pooling_ratio=pooling_ratio,
#         dropout_ratio=dropout_ratio,
#         dataset=dataset,
#         epochs=epochs,
#         patience=patience,
#         device='cpu'
#     )
    
#     # Set seeds for reproducibility
#     torch.manual_seed(args.seed)
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)

#     # Ensure deterministic behavior in cuDNN
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(args.seed)
#         args.device = 'cuda:0'
    
#     class NormalizedDegree(object):
#         def __init__(self, mean, std):
#             self.mean = mean
#             self.std = std

#         def __call__(self, data):
#             deg = degree(data.edge_index[0], dtype=torch.float)
#             deg = (deg - self.mean) / self.std
#             data.x = deg.view(-1, 1)
#             return data
        
#     # Load the dataset
#     dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True, cleaned=False)
#     dataset.data.edge_attr = None
#     max_degree = 0
#     degs = []
#     for data in dataset:
#         degs += [degree(data.edge_index[0], dtype=torch.long)]
#         max_degree = max(max_degree, degs[-1].max().item())

#     if max_degree < 1000:
#         dataset.transform = T.OneHotDegree(max_degree)
#     else:
#         deg = torch.cat(degs, dim=0).to(torch.float)
#         mean, std = deg.mean().item(), deg.std().item()
#         dataset.transform = NormalizedDegree(mean, std)

#     args.num_classes = dataset.num_classes
#     args.num_features = dataset.num_features

#     all_acc = []
#     start = time.time()

#     # Shuffle indices
#     indices = np.random.permutation(len(dataset))

#     kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)

#     for fold, (train_val_idx, test_idx) in enumerate(kf.split(indices)):
#         print(f"Fold {fold+1}/10")

#         # Split train_val into train and validation (80/10/10 split)
#         val_size = len(train_val_idx) // 9  # This gives us 10% for validation
#         val_idx = train_val_idx[:val_size]
#         train_idx = train_val_idx[val_size:]

#         train_set = Subset(dataset, indices[train_idx])
#         val_set = Subset(dataset, indices[val_idx])
#         test_set = Subset(dataset, indices[test_idx])

#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
#         val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
#         test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    
    
#         if args.netmodel == 'hire':
#             model = hireNet(args).to(args.device)
#         elif args.netmodel == 'glob':
#             model = globNet(args).to(args.device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#         model.reset_parameters()

#         # Learning rate warmup
#         def lr_lambda(epoch):
#             if epoch < 5:
#                 return (epoch + 1) / 5
#             return 1.0

#         scheduler = LambdaLR(optimizer, lr_lambda)

#         def test(model, loader):
#             model.eval()
#             correct = 0.
#             loss = 0.
#             for data in loader:
#                 data = data.to(args.device)
#                 out = model(data)
#                 pred = out.max(dim=1)[1]
#                 correct += pred.eq(data.y).sum().item()
#                 loss += F.nll_loss(out, data.y, reduction='sum').item()
#                 # loss = F.cross_entropy(out, data.y)
#             return correct / len(loader.dataset), loss / len(loader.dataset)

#         min_loss = 1e10
#         patience_counter = 0

#         for epoch in range(args.epochs):
#             model.train()
#             for i, data in enumerate(train_loader):
#                 data = data.to(args.device)
#                 out = model(data)
#                 loss = F.nll_loss(out, data.y)
#                 # loss += F.cross_entropy(out, data.y).item()
#                 loss.backward()
                
#                 # Gradient clipping
#                 clip_grad_norm_(model.parameters(), max_norm=1.0)
                
#                 optimizer.step()
#                 optimizer.zero_grad()
            
#             # Update learning rate
#             scheduler.step()
            
#             val_acc, val_loss = test(model, val_loader)
#             if val_loss < min_loss:
#                 torch.save(model.state_dict(), 'latest.pth')
#                 min_loss = val_loss
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#             if patience_counter > args.patience:
#                 break

#         model.load_state_dict(torch.load('latest.pth'))
#         test_acc, test_loss = test(model, test_loader)
#         print(f"Fold {fold+1} test accuracy: {test_acc}")
#         all_acc.append(test_acc)

#     end = time.time()
#     avg_acc = np.mean(all_acc)
#     print('Average accuracy: {:.4f}'.format(avg_acc), '+/- {:.4f}'.format(np.std(all_acc)))
#     print('Average time:', (end - start) / 10)
#     print("args: \n", args)
#     return avg_acc


# # Define the objective function for Optuna
# def objective(trial):
#     lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
#     batch_size = trial.suggest_int('batch_size', 32, 128)
#     weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
#     nhid = trial.suggest_int('nhid', 64, 2048)
#     ratio = trial.suggest_float('ratio', 0.25, 0.75)

#     return train_and_evaluate(
#         fold=10,
#         poolmodel='katzpool', #tcommpool, appnppool, powerpool, katzpool
#         convmodel='gcn',
#         netmodel='glob',
#         seed=799,     
#         batch_size=batch_size,
#         lr=lr,
#         weight_decay=weight_decay,
#         nhid=nhid,
#         K=10,
#         pooling_ratio=ratio,
#         dropout_ratio=0.5,
#         dataset='FRANKENSTEIN',
#         epochs=200,
#         patience=50
#     )

# # Run the optimization with GridSampler
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()

#     search_space = {
#         'lr': [1e-5, 1e-4],
#         'batch_size': [64],
#         'weight_decay': [1e-3],
#         'nhid': [512, 1024],
#         'ratio': [0.35, 0.55, 0.75]
#     }


#     study = optuna.create_study(direction='maximize', sampler=GridSampler(search_space))
#     study.optimize(objective, n_trials=2*2*2*2*6)

#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: {}".format(trial.value))
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))