import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
import numpy as np
from networks import  hireNet, globNet
import torch.nn.functional as F
import argparse
import os
import time
import random
from torch.utils.data import random_split
parser = argparse.ArgumentParser()

parser.add_argument('--runs', type=int, default=5,
                    help='runs')
parser.add_argument('--poolmodel', type=str, default='tcommpool',     #gcnpool, tcommpool, appnppool, powerpool, katzpool
                    help='poolmodel')
parser.add_argument('--convmodel', type=str, default='gcn',     #gcn, gat
                    help='convmodel')
parser.add_argument('--netmodel', type=str, default='hire',         #hire, glob
                    help='netmodel')
parser.add_argument('--seed', type=int, default=42,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.000001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=512,
                    help='hidden size')
parser.add_argument('--K', type=int, default=50,
                    help='number of iterative')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/ IMDB-MULTI')
parser.add_argument('--epochs', type=int, default=10000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')


args = parser.parse_args()
args.device = 'cpu'
# Set seeds for reproducibility
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Ensure deterministic behavior in cuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variable for Python hash seed
os.environ['PYTHONHASHSEED'] = str(args.seed)

# Additional settings for reproducibility in PyTorch
torch.use_deterministic_algorithms(True)  # Ensures that only deterministic algorithms are used

# Set the number of threads used by PyTorch
torch.set_num_threads(1)

# Optionally, set the number of OpenMP threads (affects NumPy and other libraries)
os.environ["OMP_NUM_THREADS"] = "1"

# Set the environment variable for CuBLAS deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'


# dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset, use_node_attr=True)
# dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset, cleaned=True)
print(dataset)
print(dataset[0])
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])


all_acc = []

start = time.time()
for run in range(args.runs):
    print("Run: ", run)

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False)

    if args.netmodel == 'hire':
        model = hireNet(args).to(args.device)
    elif args.netmodel == 'glob':
        model = globNet(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.reset_parameters()

    def test(model,loader):
        model.eval()
        correct = 0.
        loss = 0.
        for data in loader:
            data = data.to(args.device)
            out = model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss += F.nll_loss(out,data.y,reduction='sum').item()
        return correct / len(loader.dataset),loss / len(loader.dataset)


    min_loss = 1e10
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            # print("Training loss:{}".format(loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        val_acc,val_loss = test(model,val_loader)
        # print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(),'latest.pth')
            # print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break 

    if args.netmodel == 'hire':
        model = hireNet(args).to(args.device)
    elif args.netmodel == 'glob':
        model = globNet(args).to(args.device)
    model.load_state_dict(torch.load('latest.pth'))
    test_acc,test_loss = test(model,test_loader)
    print("Test accuarcy:{}".format(test_acc))
    all_acc.append(test_acc)


end = time.time()
print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))
print('ave_time:', (end-start)/args.runs)



print("args: \n", args)
