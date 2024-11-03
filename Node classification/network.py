import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch.nn import Linear
import math
from total_communicability import total_com
from Power_method import power
from katz_centrality import katz


class APPNPmodel(torch.nn.Module):
    def __init__(self, args):
        super(APPNPmodel, self).__init__()
        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)

        return F.log_softmax(x, dim=1), x



class TOTALCOMmodel(torch.nn.Module):
    def __init__(self, args):
        super(TOTALCOMmodel, self).__init__()
        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.num_classes)
        self.prop1 = total_com(args.K, 1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)

        return F.log_softmax(x, dim=1), x
    



class POWERmodel(torch.nn.Module):
    def __init__(self, args):
        super(POWERmodel, self).__init__()
        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.num_classes)
        self.prop1 = power(args.K, 1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)

        return F.log_softmax(x, dim=1), x
    

class KATZmodel(torch.nn.Module):
    def __init__(self, args):
        super(KATZmodel, self).__init__()
        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.num_classes)
        self.prop1 = power(args.K, args.alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)

        return F.log_softmax(x, dim=1), x