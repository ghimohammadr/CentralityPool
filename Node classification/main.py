import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import APPNPmodel, TOTALCOMmodel, POWERmodel, KATZmodel
import numpy as np
import time
from utils import load_data, getData
from sklearn.metrics import accuracy_score, f1_score
import os
import random
random.seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--experiment', type=str, default='fullsupervised') #'fixed', 'fullsupervised'
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--normalize_features', type=bool, default=True)
    args = parser.parse_args()
    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_acc = []
    all_macro = []

    start = time.time()
    for i in range(args.runs):

        print("Run: ", i)
        data, args.num_features, args.num_classes = getData(args.dataset)

        model = KATZmodel(args).to(device)   ### APPNPmodel, TOTALCOMmodel, POWERmodel, KATZmodel

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.reset_parameters()

        best_val_loss = float('inf')


        data, features, train_mask, val_mask, labels = load_data(args.dataset, args.experiment)
        data = data.to(device)
        features = features.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)

        if args.normalize_features:
            features = F.normalize(features, p=2)

        val_loss_history = []
        for epoch in range(args.epochs):

            model.train()
            optimizer.zero_grad()
            out, _ = model(features, data.edge_index)
            loss = F.nll_loss(out[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            pred, _ = model(features, data.edge_index)
            val_loss = F.nll_loss(pred[val_mask], labels[val_mask]).item()

            if val_loss < best_val_loss and epoch > args.epochs // 2:
                best_val_loss = val_loss
                torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')

            val_loss_history.append(val_loss)
            if args.early_stopping > 0 and epoch > args.epochs // 2:
                tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1]) 
                if val_loss > tmp.mean().item():
                    break

        model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
        model.eval()
        pred, logits = model(data.x, data.edge_index)
        data = data.cpu()
        test_acc = accuracy_score(data.y[data.test_mask], logits.argmax(1).cpu().detach().numpy()[data.test_mask])
        f1macro = f1_score(data.y[data.test_mask], logits.argmax(1).cpu().detach().numpy()[data.test_mask], average='macro')
        print("Accuracy and F1 Macro are: ", test_acc, f1macro)
        all_acc.append(test_acc)
        all_macro.append(f1macro)

    end = time.time()
    print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))
    print('ave_macro: {:.4f}'.format(np.mean(all_macro)), '+/- {:.4f}'.format(np.std(all_macro)))
    print('ave_time:', (end-start)/args.runs)






    # # Print all arguments
    # print("All arguments:")
    # for arg in vars(args):
    #     print(f"{arg}: {getattr(args, arg)}")

