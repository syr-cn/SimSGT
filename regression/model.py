import torch
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import torch.nn as nn
import copy
from graph_trans_model import TransformerNodeEncoder_v3, TransformerNodeDecoder
from torch_geometric.utils import to_dense_batch
from pos_enc.encoder import PosEncoder
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from loader_relmole import FG_NUM, fg2tensor
from loader_ecfp import ECFP_DIM, ecfp2tensor


num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), get_activation(act_func), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.add_selfloop = False

    def forward(self, x, edge_index, edge_attr):
        # #add self loops in the edge space
        if self.add_selfloop:
            edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINConv_v2(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv_v2, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), nn.BatchNorm1d(2*emb_dim), get_activation(act_func), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.add_selfloop = False
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr):
        # #add self loops in the edge space
        if self.add_selfloop:
            edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return self.activation(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINConv_v3(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv_v3, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), nn.BatchNorm1d(2*emb_dim), get_activation(act_func), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.add_selfloop = False
        self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr):
        # #add self loops in the edge space
        if self.add_selfloop:
            edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return self.activation(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINConv_v3_relu(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv_v3_relu, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), nn.BatchNorm1d(2*emb_dim), get_activation(act_func), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.add_selfloop = False
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        # #add self loops in the edge space
        if self.add_selfloop:
            edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return self.activation(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GINConvComplete(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConvComplete, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), nn.BatchNorm1d(2*emb_dim), get_activation(act_func), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding = BondEncoder(emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.edge_embedding(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GINConvComplete_v2(MessagePassing):
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConvComplete_v2, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), nn.BatchNorm1d(2*emb_dim), get_activation(act_func), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding = BondEncoder(emb_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.edge_embedding(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return self.activation(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GINConvComplete_v4(MessagePassing):
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        super(GINConvComplete_v4, self).__init__(aggr=aggr)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, out_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GINConvComplete_v3(MessagePassing):
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConvComplete_v3, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), nn.BatchNorm1d(2*emb_dim), get_activation(act_func), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding = BondEncoder(emb_dim)
        self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.edge_embedding(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return self.activation(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GINConvComplete_v3_relu(MessagePassing):
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConvComplete_v3_relu, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), nn.BatchNorm1d(2*emb_dim), get_activation(act_func), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding = BondEncoder(emb_dim)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.edge_embedding(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return self.activation(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class BondEncoder_tiny(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder_tiny, self).__init__()
        self.embedding = torch.nn.Embedding(5, emb_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, edge_attr):
        bond_embedding = self.embedding(edge_attr[:,0])
        return bond_embedding

class GINConvNode_v3(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConvNode_v3, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), nn.BatchNorm1d(2*emb_dim), get_activation(act_func), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding = BondEncoder_tiny(emb_dim)
        
        self.add_selfloop = False
        self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr):
        # #add self loops in the edge space

        edge_embeddings = self.edge_embedding(edge_attr)
        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return self.activation(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, out_dim, aggr = "add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GCNConv, self).__init__(**kwargs)

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, out_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr
        self.add_selfloop = False

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        
        if self.add_selfloop:
            #add self loops in the edge space
            edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        # x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)
        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)
    
    # added
    def update(self, aggr_out):
        return self.linear(aggr_out)

class GATConv(MessagePassing):
    def __init__(self, emb_dim, out_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out

class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr
        self.add_selfloop = False

    def forward(self, x, edge_index, edge_attr):
        if self.add_selfloop:
            #add self loops in the edge space
            edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)



class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == 'first_cat':
            node_representation = torch.cat([h_list[0], h_list[-1]], dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class TokenMAE(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("GNNTransformer - Training Config")
        ## gnn parameters
        group.add_argument('--gnn_emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
        group.add_argument('--gnn_dropout', type=float, default=0) # follow the setting of MAE
        group.add_argument('--gnn_JK', type=str, default='last')
        group.add_argument('--gnn_type', type=str, default='gin')
        group.add_argument("--gnn_activation", type=str, default="relu")
        group.add_argument("--decoder_jk", type=str, default="last")
        group.add_argument("--gnn_norm", type=str, default="batchnorm")

        ## transformer parameters
        group.add_argument('--d_model', type=int, default=128)
        group.add_argument("--dim_feedforward", type=int, default=512, help="transformer feedforward dim")
        group.add_argument("--nhead", type=int, default=4, help="transformer heads")
        group.add_argument("--transformer_dropout", type=float, default=0) # follow the setting of MAE
        group.add_argument("--transformer_activation", type=str, default="relu")
        group.add_argument("--transformer_norm_input", action="store_true", default=True)
        group.add_argument('--custom_trans', action='store_true', default=True)
        group.add_argument('--drop_mask_tokens', action='store_true', default=True)
        group.add_argument('--use_trans_decoder', action='store_true', default=False)
        # group.add_argument("--max_input_len", default=1000, help="The max input length of transformer input")
        
        ## encoder parameters
        group.add_argument('--disable_remask', action='store_true', default=False)
        group.add_argument('--tokenizer_type', type=str, default='gin')
        group.add_argument('--gnn_token_layer', type=int, default=1)
        group.add_argument('--gnn_encoder_layer', type=int, default=5)
        group.add_argument('--trans_encoder_layer', type=int, default=0)

        ## decoder parameters
        group.add_argument('--gnn_decoder_layer', type=int, default=3)
        group.add_argument('--decoder_input_norm', action='store_true', default=False)
        group.add_argument('--trans_decoder_layer', type=int, default=0)
        
        ## others
        group.add_argument('--complete_feature', action='store_true', default=False)
        group.add_argument('--nonpara_tokenizer', action='store_true', default=False)
        group.add_argument('--moving_average_decay', type=float, default=0.99)
        group.add_argument('--loss', type=str, default='mse')
        group.add_argument('--brics_pooling', type=str, default='mean')
        group.add_argument('--loss_all_nodes', action='store_true', default=False)
        group.add_argument('--subgraph_mask', action='store_true', default=False)
        group.add_argument('--zero_mask', action='store_true', default=False)
        group.add_argument('--eps', type=float, default=0.5)
        
        ## pretrained tokenizer
        group.add_argument('--pretrained_tokenizer', action='store_true', default=False)
        group.add_argument('--no_edge_tokenizer', action='store_true', default=False)
        group.add_argument('--tokenizer_path', type=str, default=None)
        group.add_argument('--tk_pretrain_scheme', type=str, default='graphmae',choices=['graphmae', 'graphcl', 'VQ-VAE'])
        group.add_argument('--tk_gnn_type', type=str, default='gin')
        group.add_argument('--tk_layers', type=int, default=1)
        group.add_argument('--tk_dropout', type=float, default=0) # follow the setting of MAE
        group.add_argument('--tk_JK', type=str, default='last')
        group.add_argument("--tk_activation", type=str, default="relu")
        group.add_argument('--tk_no_edge', action='store_true', default=False)
        group.add_argument('--tk_full_x', action='store_true', default=False)
        ### decoder
        group.add_argument('--tk_decoder_JK', type=str, default='last')
        group.add_argument('--tk_no_edge_decoder', action='store_true', default=False)
        group.add_argument('--tk_decoder_layers', type=int, default=1)
        group.add_argument('--tk_decoder_remask', action='store_true', default=False)
        ### graph transformer
        group.add_argument('--tk_trans_layers', type=int, default=0)

        ## PE Config
        group_pe = parser.add_argument_group("PE Config")
        group_pe.add_argument('--pe_type', type=str, default='none',choices=['none', 'signnet', 'lap', 'lap_v2', 'signnet_v2', 'rwse', 'signnet_v3'])
        group_pe.add_argument('--laplacian_norm', type=str, default='none')
        group_pe.add_argument('--max_freqs', type=int, default=20)
        group_pe.add_argument('--eigvec_norm', type=str, default='L2')
        group_pe.add_argument('--raw_norm_type', type=str, default='none', choices=['none', 'batchnorm'])
        group_pe.add_argument('--kernel_times', type=list, default=[]) # cmd line param not supported yet
        group_pe.add_argument('--kernel_times_func', type=str, default='none')
        group_pe.add_argument('--layers', type=int, default=3)
        group_pe.add_argument('--post_layers', type=int, default=2)
        group_pe.add_argument('--dim_pe', type=int, default=28, help='dim of node positional encoding')
        group_pe.add_argument('--phi_hidden_dim', type=int, default=32)
        group_pe.add_argument('--phi_out_dim', type=int, default=32)

    def __init__(self, gnn_encoder_layer, gnn_token_layer, gnn_decoder_layer, gnn_emb_dim, nonpara_tokenizer=False, gnn_JK = "last", gnn_dropout = 0, gnn_type = "gin",
    d_model=128, trans_encoder_layer=0, trans_decoder_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False, use_trans_decoder=False, pe_type='none', args=None):
        super().__init__()
        assert gnn_JK == 'last'
        self.pe_type = pe_type
        self.loss_all_nodes = args.loss_all_nodes
        self.loss = args.loss
        self.pos_encoder = PosEncoder(args)
        if args.disable_remask:
            drop_mask_tokens=False
        if args.complete_feature:
            self.tokenizer = GNNComplete_v2(1, gnn_emb_dim, True, JK=gnn_JK, drop_ratio=gnn_dropout, gnn_type=gnn_type, gnn_activation=args.gnn_activation, gnn_norm=args.gnn_norm)
            self.encoder = GNNComplete_v2(gnn_encoder_layer-1, gnn_emb_dim, False, JK=gnn_JK, drop_ratio=gnn_dropout, gnn_type=gnn_type, gnn_activation=args.gnn_activation, gnn_norm=args.gnn_norm, 
                d_model=d_model, trans_layer=trans_encoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens, pe_dim=self.pos_encoder.pe_dim)
        else:
            self.tokenizer = GNN_v2(1, gnn_emb_dim, True, JK=gnn_JK, drop_ratio=gnn_dropout, gnn_type=gnn_type, gnn_activation=args.gnn_activation)
            self.encoder = GNN_v2(gnn_encoder_layer-1, gnn_emb_dim, False, JK=gnn_JK, drop_ratio=gnn_dropout, gnn_type=gnn_type, gnn_activation=args.gnn_activation, 
                d_model=d_model, trans_layer=trans_encoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens, pe_dim=self.pos_encoder.pe_dim)

        self.gnn_act = get_activation(args.gnn_activation)
        self.nonpara_tokenizer = nonpara_tokenizer
        self.pretrained_tokenizer = args.pretrained_tokenizer
        self.remask = not args.disable_remask

        self.mask_embed = nn.Parameter(torch.zeros(gnn_emb_dim))
        nn.init.normal_(self.mask_embed, std=.02)

        if args.tk_JK == 'concat':
            out_dim = gnn_emb_dim * (args.tk_layers + 1)
        elif gnn_token_layer == 0:
            out_dim = num_atom_type
        elif args.tk_trans_layers >0:
            out_dim = d_model
        else:
            out_dim = gnn_emb_dim

        if self.pretrained_tokenizer:
            if 'mae' in args.tokenizer_path:
                if args.no_edge_tokenizer:
                    self.tokenizer_pretrained = NoEdgeTokenizer(
                        args.tk_layers,
                        args.gnn_emb_dim,
                        JK = args.tk_JK,
                        gnn_type = args.tk_gnn_type
                    )
                elif args.tk_trans_layers > 0:
                    self.tokenizer_pretrained = transTokenizer(
                        args.tk_layers,
                        args.gnn_emb_dim,
                        JK = args.tk_JK,
                        gnn_type = args.tk_gnn_type,
                        trans_layer = args.tk_trans_layers,
                        custom_trans = True,
                        transformer_norm_input = True
                    )
                else:
                    self.tokenizer_pretrained = ParaTokenizer(args)
                self.tokenizer_pretrained.from_pretrained(args.tokenizer_path)
            elif 'gcl' in args.tokenizer_path:
                from model_graphcl import GNN as GNN_graphcl
                self.tokenizer_pretrained = GNN_graphcl(args.tk_layers, args.gnn_emb_dim, with_edge=not args.no_edge_tokenizer)
                state_dict = torch.load(args.tokenizer_path, map_location=torch.device('cpu'))
                self.tokenizer_pretrained.load_state_dict(state_dict)
            elif 'vae' in args.tokenizer_path:
                if args.no_edge_tokenizer:
                    self.tokenizer_pretrained = NoEdgeTokenizer(
                        args.tk_layers,
                        args.gnn_emb_dim,
                        JK = args.tk_JK,
                        gnn_type = args.tk_gnn_type
                    )
                else:
                    self.tokenizer_pretrained = ParaTokenizer(args)
                vq_num_tokens = 512 # hard coded in vq-vae's code
                self.tokenizer_pretrained.from_pretrained(args.tokenizer_path)
                self.vq_layer = VectorQuantizer(out_dim, vq_num_tokens, 1)
                vq_path = args.tokenizer_path.replace('.pth', '_vq.pth')
                self.vq_layer.from_pretrained(vq_path)
                self.probe = nn.Linear(out_dim, vq_num_tokens)

        elif self.nonpara_tokenizer:
            self.tokenizer_nonpara = Tokenizer(gnn_emb_dim, gnn_token_layer, args.eps, JK=gnn_JK, gnn_type=args.tokenizer_type)

        if trans_encoder_layer > 0:
            in_dim = d_model
        else:
            in_dim = gnn_emb_dim
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.brics_tokenizer = 'brics' in args.dataset
        if self.brics_tokenizer:
            VOCAB_SIZE = 12233 # hard coded vocab size of clique.txt
            self.brics_pooling = args.brics_pooling
            self.probe = nn.Linear(out_dim, VOCAB_SIZE)
        self.ecfp_tokenizer = 'ecfp' in args.dataset
        if self.ecfp_tokenizer:
            self.probe = nn.Linear(out_dim, ECFP_DIM)
        self.relmole_tokenizer = 'relmole' in args.dataset
        if self.relmole_tokenizer:
            self.probe = nn.Linear(out_dim, FG_NUM)


        self.use_trans_decoder = use_trans_decoder
        if self.use_trans_decoder:
            in_dim = d_model + self.pos_encoder.pe_dim
            self.decoder = TransDecoder(in_dim, out_dim, d_model=d_model, trans_layer=trans_decoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens)
            # self.decoder = GNNDecoder_v3(in_dim, gnn_emb_dim, gnn_emb_dim, gnn_decoder_layer, gnn_type=gnn_type, 
            # d_model=d_model, trans_layer=trans_decoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens)
        else:
            if args.complete_feature:
                self.decoder = GNNDecoderComplete_v2(in_dim, gnn_emb_dim, out_dim, gnn_decoder_layer, gnn_type=gnn_type, gnn_activation=args.gnn_activation, gnn_jk=args.decoder_jk,
                d_model=d_model, trans_layer=trans_decoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens and trans_encoder_layer > 0, pe_dim=self.pos_encoder.pe_dim, use_input_norm=args.decoder_input_norm, zero_mask=args.zero_mask)
            else:
                self.decoder = GNNDecoder_v2(in_dim, gnn_emb_dim, out_dim, gnn_decoder_layer, gnn_type=gnn_type, gnn_activation=args.gnn_activation, gnn_jk=args.decoder_jk,
                d_model=d_model, trans_layer=trans_decoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens and trans_encoder_layer > 0, pe_dim=self.pos_encoder.pe_dim, use_input_norm=args.decoder_input_norm, zero_mask=args.zero_mask)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        ## forward tokenizer
        h = self.tokenizer(data.x_masked, edge_index, edge_attr)
        
        ## forward tokenizer target
        with torch.no_grad():
            if self.pretrained_tokenizer:
                if hasattr(self.tokenizer_pretrained, 'trans_enc'):
                    g_tokens = self.tokenizer_pretrained(x, edge_index, edge_attr, data.batch).detach()
                else:
                    g_tokens = self.tokenizer_pretrained(x, edge_index, edge_attr).detach()
                if hasattr(self, 'vq_layer'):
                    g_tokens = self.vq_layer.get_code_indices(x, g_tokens)
            elif self.brics_tokenizer:
                if self.brics_pooling=='none':
                    motif_batch = None
                    motif_id = data.vocab_id
                else:
                    HASH_BASE = int(1e6)
                    hash_id = HASH_BASE*data.vocab_id + data.batch
                    motif_id, motif_batch = hash_id.unique(dim=0, return_inverse=True)
                    motif_id = torch.floor_divide(motif_id, HASH_BASE)

                    # original design ↓
                    # motif_id, motif_batch = data.vocab_id.unique(dim=0, return_inverse=True)
                # motif_id = motif_id[:, 0]
                g_tokens = motif_id
            elif self.ecfp_tokenizer:
                g_tokens = ecfp2tensor(data.ecfp4).to(x.device)
            elif self.relmole_tokenizer:
                g_tokens = fg2tensor(data.vocab_id).to(x.device)
            elif self.nonpara_tokenizer:
                if hasattr(self.tokenizer, 'x_embedding1'):
                    g_tokens = self.tokenizer_nonpara(x, edge_index, self.tokenizer.x_embedding1).detach()
                else:
                    g_tokens = self.tokenizer_nonpara(x, edge_index, self.tokenizer.x_embedding.atom_embedding_list[0]).detach()
            else:
                g_tokens = self.tokenizer(x, edge_index, edge_attr).detach()

        pe_tokens = self.pos_encoder(data)

        # forward encoder
        h = self.encoder(self.gnn_act(h), edge_index, edge_attr, data.batch, data.mask_tokens, pe_tokens)

        ## forward decoder
        if self.use_trans_decoder:
            g_pred = self.decoder(h, pe_tokens, data.mask_tokens, data.batch, remask=self.remask)
        else:
            g_pred = self.decoder(h, edge_index, edge_attr, data.mask_tokens, data.batch, pe_tokens, remask=self.remask)

        if self.brics_tokenizer and (motif_batch is not None):
            self.loss_all_nodes = True
            global_pool = {
                'mean':global_mean_pool,
                'max':global_max_pool,
                'sum':global_add_pool,
            }[self.brics_pooling]
            g_pred = global_pool(g_pred, motif_batch)

        ## compute loss
        if not self.loss_all_nodes:
            g_pred = g_pred[data.mask_tokens]
            g_tokens = g_tokens[data.mask_tokens]

        if self.loss == 'mse':
            loss = self.mse_loss(g_tokens, g_pred)
        elif self.loss == 'rmse':
            loss = self.rmse_loss(g_tokens, g_pred)
        elif self.loss == 'sce':
            loss = self.sce_loss(g_tokens, g_pred)
        elif self.loss == 'ce':
            g_pred = self.probe(g_pred)
            loss = self.ce_loss(g_pred, g_tokens)
        else:
            raise NotImplementedError()
        return loss


    def mse_loss(self, x, y):
        loss = ((x - y) ** 2).mean()
        return loss
    
    def rmse_loss(self, x, y):
        loss = sqrt(((x - y) ** 2).mean())
        return loss

    def sce_loss(self, x, y, alpha: float=1):
        x = F.normalize(x, p=2.0, dim=-1) # shape = [N, D]
        y = F.normalize(y, p=2.0, dim=-1) # shape = [N, D]
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    @torch.no_grad()
    def update_tokenizer(self, momentum):
        for current_params, ma_params in zip(self.tokenizer.parameters(), self.tokenizer_ema.parameters()):
            up_weight, old_weight = current_params.data, ma_params.data
            ma_params.data = (1 - momentum) * up_weight + momentum * old_weight


class NonParaGINConv(MessagePassing):
    ## non-parametric gin
    def __init__(self, eps, aggr = "add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)
        self.aggr = aggr
        self.eps = eps

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x) + x * self.eps

    def message(self, x_j):
        return x_j

class ParaGINConv(MessagePassing):
    ## non-parametric gin
    def __init__(self, emb_dim, out_dim, aggr = "add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, out_dim))
        self.aggr = aggr
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, _):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class NonParaGCNConv(MessagePassing):
    ## non-parametric gcn
    def __init__(self, eps, aggr = "add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)
        self.aggr = aggr
        self.eps = eps

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index):
        norm = self.norm(edge_index, x.size(0), x.dtype)
        return self.propagate(edge_index, x=x, norm=norm) + x * self.eps

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class ParaGCNConv(MessagePassing):
    def __init__(self, emb_dim, out_dim, aggr = "add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(ParaGCNConv, self).__init__(**kwargs)
        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, out_dim)
        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        norm = self.norm(edge_index, x.size(0), x.dtype)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * (x_j)
    
    # added
    def update(self, aggr_out):
        return self.linear(aggr_out)


class NonParaGraphSAGEConv(MessagePassing):
    ## non-parametric gcn
    def __init__(self, eps, aggr = "add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)
        self.aggr = aggr
        self.eps = eps

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x) + x * self.eps

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)

class Tokenizer(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, emb_dim, num_layer, eps, JK = "last", gnn_type = "gin"):
        super().__init__()
        self.num_layer = num_layer
        self.JK = JK
        
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(NonParaGINConv(eps))
            elif gnn_type == "gcn":
                self.gnns.append(NonParaGCNConv(eps))
            elif gnn_type == "graphsage":
                self.gnns.append(NonParaGraphSAGEConv(eps, aggr='mean'))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim, affine=False))

    def forward(self, x, edge_index, node_embedding):
        if self.num_layer == 0:
            return F.one_hot(x[:, 0], num_classes=num_atom_type).float()
        x = node_embedding(x[:, 0])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            g_tokens = torch.cat(h_list, dim = 1)
        elif self.JK == 'first_cat':
            g_tokens = torch.cat([h_list[0], h_list[-1]], dim = 1)
        elif self.JK == "last":
            g_tokens = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            g_tokens = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            g_tokens = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
        return g_tokens



class ParaTokenizer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        GNNEncoder = GNNComplete_v2 if args.complete_feature else GNN_v2
        self.gnn = GNNEncoder(
            args.tk_layers,
            args.gnn_emb_dim,
            True,
            JK=args.tk_JK,
            drop_ratio=args.tk_dropout,
            gnn_type=args.tk_gnn_type,
            gnn_activation=args.tk_activation
        )

    def forward(self, x, edge_index, edge_attr):
        return self.gnn(x, edge_index, edge_attr)

    def from_pretrained(self, path):
        assert path
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)

class NoEdgeTokenizer(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(NoEdgeTokenizer, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(ParaGINConv(emb_dim, emb_dim, aggr = "add"))
            if gnn_type == "gcn":
                self.gnns.append(ParaGCNConv(emb_dim, emb_dim, aggr = "add"))
            else:
                raise NotImplementedError(f"{gnn_type}")
        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0])# + self.x_embedding2(x[:,1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.load_state_dict(torch.load(model_file))


class transTokenizer(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, input_layer=True, JK = "last", drop_ratio = 0, gnn_type = "gin", gnn_activation='relu',
    d_model=128, trans_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False, pe_dim=0, trans_pooling='none'):
        super(transTokenizer, self).__init__()
        self.trans_pooling = trans_pooling
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.drop_mask_tokens = drop_mask_tokens

        self.input_layer = input_layer
        if self.input_layer:
            self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        self.activations = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(ParaGINConv(emb_dim, emb_dim, aggr = "add"))
            else:
                raise NotImplementedError()
            self.activations.append(get_activation(gnn_activation))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        self.trans_layer = trans_layer
        if self.trans_layer > 0:
            self.gnn2trans = nn.Linear(emb_dim+pe_dim, d_model, bias=False)
            self.gnn2trans_act = get_activation(gnn_activation)
            self.trans_enc = TransformerNodeEncoder_v3(d_model, trans_layer, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans=custom_trans)


    #def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.x_embedding1(x[:,0])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(self.activations[layer](h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == 'first_cat':
            node_representation = torch.cat([h_list[0], h_list[-1]], dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        if self.trans_layer > 0:
            node_representation = self.gnn2trans_act(self.gnn2trans(node_representation))
            pad_x, pad_mask = to_dense_batch(node_representation, batch) # shape = [B, N_max, D], shape = [B, N_max]
            pad_x = pad_x.permute(1, 0, 2)
            pad_x, _ = self.trans_enc(pad_x, ~pad_mask) # discard the cls token; shape = [N_max+1, B, D]
            if self.trans_pooling == 'cls':
                return pad_x[-1]
            pad_x = pad_x[:-1] # discard the cls token; shape = [N_max, B, D]
            node_representation = pad_x.permute(1, 0, 2)[pad_mask]
            
        return node_representation
    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.load_state_dict(torch.load(model_file))


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
    def forward(self, x, e):    
        encoding_indices = self.get_code_indices(x, e) # x: B * H, encoding_indices: B
        quantized = self.quantize(encoding_indices)
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, e.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(e, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # Straight Through Estimator
        quantized = e + (quantized - e).detach().contiguous()
        return quantized, loss
    
    def get_code_indices(self, x, e):
        # x: N * 2  e: N * E
        atom_type = x[:, 0]
        index_c = (atom_type == 5)
        index_n = (atom_type == 6)
        index_o = (atom_type == 7)
        index_others = ~(index_c + index_n + index_o)
        # compute L2 distance
        encoding_indices = torch.ones(e.size(0)).long().to(e.device)
        # C:
        distances = (
            torch.sum(e[index_c] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[0: 377] ** 2, dim=1) -
            2. * torch.matmul(e[index_c], self.embeddings.weight[0: 377].t())
        )
        encoding_indices[index_c] = torch.argmin(distances, dim=1)
        # N:
        distances = (
            torch.sum(e[index_n] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[378: 433] ** 2, dim=1) -
            2. * torch.matmul(e[index_n], self.embeddings.weight[378: 433].t())
        ) 
        encoding_indices[index_n] = torch.argmin(distances, dim=1) + 378
        # O:
        distances = (
            torch.sum(e[index_o] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[434: 488] ** 2, dim=1) -
            2. * torch.matmul(e[index_o], self.embeddings.weight[434: 488].t())
        )   
        encoding_indices[index_o] = torch.argmin(distances, dim=1) + 434

        # Others:
        distances = (
            torch.sum(e[index_others] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[489: 511] ** 2, dim=1) -
            2. * torch.matmul(e[index_others], self.embeddings.weight[489: 511].t())
        ) 
        encoding_indices[index_others] = torch.argmin(distances, dim=1) + 489
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices) 

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))    

def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()
    else:
        raise NotImplementedError()


class GNN_v2(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, input_layer=True, JK = "last", drop_ratio = 0, gnn_type = "gin", gnn_activation='relu',
    d_model=128, trans_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False, pe_dim=0, trans_pooling='none'):
        super(GNN_v2, self).__init__()
        self.trans_pooling = trans_pooling
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.drop_mask_tokens = drop_mask_tokens

        # if self.num_layer < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        self.input_layer = input_layer
        if self.input_layer:
            self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
            self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        self.activations = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v2':
                self.gnns.append(GINConv_v2(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v3':
                self.gnns.append(GINConv_v3(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v3_relu':
                self.gnns.append(GINConv_v3_relu(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, emb_dim, aggr = "add"))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, aggr = "add"))
            else:
                raise NotImplementedError()
            self.activations.append(get_activation(gnn_activation))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        self.trans_layer = trans_layer
        if self.trans_layer > 0:
            self.gnn2trans = nn.Linear(emb_dim+pe_dim, d_model, bias=False)
            self.gnn2trans_act = get_activation(gnn_activation)
            self.trans_enc = TransformerNodeEncoder_v3(d_model, trans_layer, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans=custom_trans)


    #def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr, batch=None, mask_tokens=None, pe_tokens=None):
        if self.input_layer:
            x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(self.activations[layer](h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == 'first_cat':
            node_representation = torch.cat([h_list[0], h_list[-1]], dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]


        ## add pe tokens
        if pe_tokens is not None:
            node_representation = torch.cat((node_representation, pe_tokens), dim=-1)

        if self.trans_layer > 0:
            assert batch is not None
            if self.drop_mask_tokens and (mask_tokens is not None):
                assert mask_tokens is not None
                unmask_tokens = ~mask_tokens
                node_representation = node_representation[unmask_tokens]
                node_representation = self.gnn2trans_act(self.gnn2trans(node_representation))
                pad_x, pad_mask = to_dense_batch(node_representation, batch[unmask_tokens]) # shape = [B, N_max, D], shape = [B, N_max]
                pad_x = pad_x.permute(1, 0, 2)
                pad_x, _ = self.trans_enc(pad_x, ~pad_mask) # discard the cls token; shape = [N_max+1, B, D]
                if self.trans_pooling == 'cls':
                    return pad_x[-1]
                pad_x = pad_x[:-1] # discard the cls token; shape = [N_max, B, D]
                node_representation = pad_x.permute(1, 0, 2)[pad_mask]
            else:
                node_representation = self.gnn2trans_act(self.gnn2trans(node_representation))
                pad_x, pad_mask = to_dense_batch(node_representation, batch) # shape = [B, N_max, D], shape = [B, N_max]
                pad_x = pad_x.permute(1, 0, 2)
                pad_x, _ = self.trans_enc(pad_x, ~pad_mask) # discard the cls token; shape = [N_max+1, B, D]
                if self.trans_pooling == 'cls':
                    return pad_x[-1]
                pad_x = pad_x[:-1] # discard the cls token; shape = [N_max, B, D]
                node_representation = pad_x.permute(1, 0, 2)[pad_mask]
            
        return node_representation


class GNNComplete_v2(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, input_layer=True, JK = "last", drop_ratio = 0, gnn_type = "gin", gnn_activation='relu', gnn_norm='batchnorm',
    d_model=128, trans_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False, pe_dim=0, trans_pooling='none'):
        super(GNNComplete_v2, self).__init__()
        self.trans_pooling = trans_pooling
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.drop_mask_tokens = drop_mask_tokens

        # if self.num_layer < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        self.input_layer = input_layer
        if self.input_layer:
            self.x_embedding = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        self.activations = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.gnns.append(GINConvComplete(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v2':
                self.gnns.append(GINConvComplete_v2(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v3':
                self.gnns.append(GINConvComplete_v3(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v3_relu':
                self.gnns.append(GINConvComplete_v3_relu(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v4':
                self.gnns.append(GINConvComplete_v4(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            else:
                raise NotImplementedError()
            self.activations.append(get_activation(gnn_activation))

        ###List of batchnorms
        if gnn_norm == 'l2norm':
            self.batch_norms = []
        else:
            self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_norm == 'batchnorm':
                self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            elif gnn_norm == 'layernorm':
                self.batch_norms.append(torch.nn.LayerNorm(emb_dim))
            elif gnn_norm == 'l2norm':
                self.batch_norms.append(lambda x:nn.functional.normalize(x, p = 2, dim = -1))
            else:
                raise NotImplementedError(f'gnn normalize method {gnn_norm} not implemented!')

        self.trans_layer = trans_layer
        if self.trans_layer > 0:
            self.gnn2trans = nn.Linear(emb_dim+pe_dim, d_model, bias=False)
            self.gnn2trans_act = get_activation(gnn_activation)
            self.trans_enc = TransformerNodeEncoder_v3(d_model, trans_layer, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans=custom_trans)


    #def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr, batch=None, mask_tokens=None, pe_tokens=None):
        if self.input_layer:
            x = self.x_embedding(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(self.activations[layer](h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == 'first_cat':
            node_representation = torch.cat([h_list[0], h_list[-1]], dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]


        ## add pe tokens
        if pe_tokens is not None:
            node_representation = torch.cat((node_representation, pe_tokens), dim=-1)

        if self.trans_layer > 0:
            assert batch is not None
            if self.drop_mask_tokens:
                assert mask_tokens is not None
                unmask_tokens = ~mask_tokens
                node_representation = node_representation[unmask_tokens]
                node_representation = self.gnn2trans_act(self.gnn2trans(node_representation))
                pad_x, pad_mask = to_dense_batch(node_representation, batch[unmask_tokens]) # shape = [B, N_max, D], shape = [B, N_max]
                pad_x = pad_x.permute(1, 0, 2)
                pad_x, _ = self.trans_enc(pad_x, ~pad_mask) # discard the cls token; shape = [N_max+1, B, D]
                if self.trans_pooling == 'cls':
                    return pad_x[-1]
                pad_x = pad_x[:-1] # discard the cls token; shape = [N_max, B, D]
                node_representation = pad_x.permute(1, 0, 2)[pad_mask]
            else:
                node_representation = self.gnn2trans_act(self.gnn2trans(node_representation))
                pad_x, pad_mask = to_dense_batch(node_representation, batch) # shape = [B, N_max, D], shape = [B, N_max]
                pad_x = pad_x.permute(1, 0, 2)
                pad_x, _ = self.trans_enc(pad_x, ~pad_mask) # discard the cls token; shape = [N_max+1, B, D]
                if self.trans_pooling == 'cls':
                    return pad_x[-1]
                pad_x = pad_x[:-1] # discard the cls token; shape = [N_max, B, D]
                node_representation = pad_x.permute(1, 0, 2)[pad_mask]
            
        return node_representation


class GNN_v3(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, input_layer=True, JK = "last", drop_ratio = 0, gnn_type = "gin", gnn_activation='relu',
    d_model=128, trans_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False, pe_dim=0, trans_pooling='none'):
        super(GNN_v3, self).__init__()
        self.trans_pooling = trans_pooling
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.drop_mask_tokens = drop_mask_tokens

        # if self.num_layer < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        self.input_layer = input_layer
        if self.input_layer:
            self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
            self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        self.activations = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v2':
                self.gnns.append(GINConv_v2(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v3':
                self.gnns.append(GINConv_v3(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v3_relu':
                self.gnns.append(GINConv_v3_relu(emb_dim, emb_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))
            else:
                raise NotImplementedError()
            self.activations.append(get_activation(gnn_activation))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        self.trans_layer = trans_layer
        if self.trans_layer > 0:
            self.gnn2trans = nn.Linear(emb_dim+pe_dim, d_model, bias=False)
            self.gnn2trans_act = get_activation(gnn_activation)
            self.trans_enc = TransformerNodeEncoder_v3(d_model, trans_layer, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans=custom_trans)


    #def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr, batch=None, mask_tokens=None, pe_tokens=None, extra_feat=None, extra_edge_attr=[None], disable_trans=False):
        if self.input_layer:
            x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
            if extra_feat is not None:
                x = x + extra_feat

        h_list = [x]
        for layer in range(self.num_layer):
            if extra_edge_attr[0] is not None:
                h = self.gnns[layer](h_list[layer], edge_index, edge_attr, extra_edge_attr[layer])
            else:
                h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(self.activations[layer](h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == 'first_cat':
            node_representation = torch.cat([h_list[0], h_list[-1]], dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
        
        if disable_trans:
            return node_representation

        ## add pe tokens
        if pe_tokens is not None:
            node_representation = torch.cat((node_representation, pe_tokens), dim=-1)

        if self.trans_layer > 0:
            assert batch is not None
            if self.drop_mask_tokens:
                assert mask_tokens is not None
                unmask_tokens = ~mask_tokens
                node_representation = node_representation[unmask_tokens]
                node_representation = self.gnn2trans_act(self.gnn2trans(node_representation))
                pad_x, pad_mask = to_dense_batch(node_representation, batch[unmask_tokens]) # shape = [B, N_max, D], shape = [B, N_max]
                pad_x = pad_x.permute(1, 0, 2)
                pad_x, _ = self.trans_enc(pad_x, ~pad_mask) # discard the cls token; shape = [N_max+1, B, D]
                if self.trans_pooling == 'cls':
                    return pad_x[-1]
                pad_x = pad_x[:-1] # discard the cls token; shape = [N_max, B, D]
                node_representation = pad_x.permute(1, 0, 2)[pad_mask]
            else:
                node_representation = self.gnn2trans_act(self.gnn2trans(node_representation))
                pad_x, pad_mask = to_dense_batch(node_representation, batch) # shape = [B, N_max, D], shape = [B, N_max]
                pad_x = pad_x.permute(1, 0, 2)
                pad_x, _ = self.trans_enc(pad_x, ~pad_mask) # discard the cls token; shape = [N_max+1, B, D]
                if self.trans_pooling == 'cls':
                    return pad_x[-1]
                pad_x = pad_x[:-1] # discard the cls token; shape = [N_max, B, D]
                node_representation = pad_x.permute(1, 0, 2)[pad_mask]
            
        return node_representation


class GNNDecoder_v3(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, gnn_layer=1, drop_ratio = 0, gnn_type = "gin", 
    d_model=128, trans_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False):
        super().__init__()
        assert hidden_dim == out_dim
        self.num_layer = gnn_layer
        self.drop_mask_tokens = drop_mask_tokens
        self.gnns = torch.nn.ModuleList()
        for layer in range(gnn_layer-1):
            if gnn_type == "gin":
                self.gnns.append(GINConv(hidden_dim, hidden_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(hidden_dim, hidden_dim, aggr = "add"))
            elif gnn_type == "linear":
                self.gnns.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                raise NotImplementedError(f"{gnn_type}")
            
        if gnn_type == "gin":
            self.gnns.append(GINConv(hidden_dim, out_dim, aggr = "add"))
        elif gnn_type == "gcn":
            self.gnns.append(GCNConv(hidden_dim, out_dim, aggr = "add"))
        elif gnn_type == "linear":
            self.gnns.append(nn.Linear(hidden_dim, out_dim))
        else:
            raise NotImplementedError(f"{gnn_type}")
        
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(drop_ratio)
        self.enc_to_dec = torch.nn.Linear(in_dim, hidden_dim, bias=False)

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(gnn_layer-1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))
        self.use_mask_emb = True
        self.mask_embed = nn.Parameter(torch.zeros((1, hidden_dim,)))
        nn.init.normal_(self.mask_embed, std=.02)

        self.trans_layer = trans_layer
        if self.trans_layer > 0:
            self.gnn2trans = nn.Linear(hidden_dim, d_model, bias=False)
            self.trans2out = nn.Linear(d_model, out_dim, bias=False)
            self.trans_decoder = TransformerNodeDecoder(d_model, trans_layer, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans=custom_trans)
            self.memory2decoder = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x, edge_index, edge_attr, masked_tokens, batch):
        x = self.activation(x)
        x = self.enc_to_dec(x)
        
        unmask_tokens = ~masked_tokens
        if self.use_mask_emb:
            # x[mask_node_indices] = self.mask_embed
            if self.drop_mask_tokens:
                ## get memory
                memory = x
                memory_batch = batch[unmask_tokens]

                ## recover the masked tokens
                box = self.mask_embed.repeat(batch.shape[0], 1)
                box[~masked_tokens] = x
                x = box
            else:
                ## get memory
                memory = x[unmask_tokens]
                memory_batch = batch[unmask_tokens]

                ## re-mask    
                x = torch.where(masked_tokens.reshape(-1, 1), self.mask_embed, x)
        else:
            ## get memory
            memory = x[unmask_tokens]
            memory_batch = batch[unmask_tokens]

            ## re-mask
            x[masked_tokens] = 0

        for layer in range(self.num_layer):
            x = self.gnns[layer](x, edge_index, edge_attr)
            x = self.batch_norms[layer](x)
            if layer != self.num_layer - 1:
                x = F.relu(x)
            x = self.dropout(x)
        
        if self.trans_layer > 0:
            x = F.relu(self.gnn2trans(x))
            memory = self.memory2decoder(memory)
            assert batch is not None
            pad_x, pad_mask = to_dense_batch(x, batch) # shape = [B, N_max, D], shape = [B, N_max]
            pad_memory, pad_memory_mask = to_dense_batch(memory, memory_batch) # shape = [B, N_max, D], shape = [B, N_max]

            pad_x = pad_x.permute(1, 0, 2)
            pad_memory = pad_memory.permute(1, 0, 2)

            pad_out = self.trans_decoder(pad_x, pad_memory, ~pad_mask, ~pad_memory_mask) # discard the cls token; shape = [N_max+1, B, D]
            trans_out = pad_out.permute(1, 0, 2)[pad_mask]
            trans_out = self.trans2out(trans_out)
            x = trans_out

        return x


class TransDecoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, d_model=128, trans_layer=2, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False):
        super().__init__()
        assert trans_layer > 0
        assert drop_mask_tokens

        self.activation = nn.PReLU()
        self.enc_to_dec = torch.nn.Linear(in_dim, d_model)
        ###List of batchnorms
        self.mask_embed = nn.Parameter(torch.zeros((1, d_model)))
        nn.init.normal_(self.mask_embed, std=.02)
        
        self.trans2out = nn.Linear(d_model, out_dim, bias=False)
        self.trans_decoder = TransformerNodeEncoder_v3(d_model, trans_layer, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans=custom_trans)

    def forward(self, x, pos_enc, masked_tokens, batch):
        '''
        x: shape = 
        '''
        ## recover masked nodes
        box = self.mask_embed.repeat(batch.shape[0], 1)
        box[~masked_tokens] = x
        x = box
        
        ## cat pos_enc
        x = torch.cat((x, pos_enc), dim=-1) # shape = [N, d_model + pe_dim]
        x = self.enc_to_dec(x)
        x = self.activation(x)
        
        ## forward transformer encoder for decoding
        pad_x, pad_mask = to_dense_batch(x, batch) # shape = [B, N_max, D], shape = [B, N_max]
        pad_x = pad_x.permute(1, 0, 2)
        pad_out, _ = self.trans_decoder(pad_x, ~pad_mask) # discard the cls token; shape = [N_max+1, B, D]
        pad_out = pad_out[:-1] # discard the cls token; shape = [N_max, B, D]
        trans_out = pad_out.permute(1, 0, 2)[pad_mask]
        trans_out = self.trans2out(trans_out)
        return trans_out


class MaskGNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(MaskGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        self.mask_embed = nn.Parameter(torch.zeros(emb_dim))
        nn.init.normal_(self.mask_embed, std=.02)


    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]

        ## the first layer
        h = self.gnns[0](h_list[0], edge_index, edge_attr)
        h = self.batch_norms[0](h)
        
        ## get the target graph tokens
        g_tokens = h.detach()

        
        h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
        
        # conduct masking
        h = torch.where(data.mask_tokens.reshape(-1, 1), self.mask_embed.reshape(1, -1), h)

        h_list.append(h)

        ##  the rest layers
        for layer in range(1, self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == 'first_cat':
            node_representation = torch.cat([h_list[0], h_list[-1]], dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation, g_tokens


class GNNDecoder_v2(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, gnn_layer=1, drop_ratio = 0, gnn_type = "gin", gnn_activation='relu', gnn_jk='last',
    d_model=128, trans_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False, pe_dim=0, use_input_norm=False, zero_mask=False):
        super().__init__()
        self.gnn_jk = gnn_jk
        self.num_layer = gnn_layer
        self.drop_mask_tokens = drop_mask_tokens
        self.gnns = nn.ModuleList()
        self.activations = nn.ModuleList()
        for layer in range(gnn_layer-1):
            if gnn_type == "gin":
                self.gnns.append(GINConv(hidden_dim, hidden_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v2':
                self.gnns.append(GINConv_v2(hidden_dim, hidden_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v3':
                self.gnns.append(GINConv_v3(hidden_dim, hidden_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v4':
                self.gnns.append(GINConv_v3(hidden_dim, hidden_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(hidden_dim, hidden_dim, aggr = "add"))
            elif gnn_type == "linear":
                self.gnns.append(nn.Linear(hidden_dim, hidden_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(hidden_dim, aggr = "add"))
            else:
                raise NotImplementedError(f"{gnn_type}")
            self.activations.append(get_activation(gnn_activation))
        
        if trans_layer > 0:
            next_dim = hidden_dim
        else:
            if gnn_jk == 'concat':
                self.combine = nn.Linear(hidden_dim * gnn_layer, out_dim)
                next_dim = hidden_dim
            elif gnn_jk == 'last':
                next_dim = out_dim
            else:
                raise NotImplementedError()

        if gnn_type == "gin":
            self.gnns.append(GINConv(hidden_dim, next_dim, aggr = "add", act_func=gnn_activation))
        elif gnn_type == 'gin_v2':
            self.gnns.append(GINConv_v2(hidden_dim, next_dim, aggr = "add", act_func=gnn_activation))
        elif gnn_type == 'gin_v3':
            self.gnns.append(GINConv_v3(hidden_dim, next_dim, aggr = "add", act_func=gnn_activation))
        elif gnn_type == "gcn":
            self.gnns.append(GCNConv(hidden_dim, next_dim, aggr = "add"))
        elif gnn_type == "graphsage":
            self.gnns.append(GraphSAGEConv(hidden_dim, aggr = "add"))
        elif gnn_type == "linear":
            self.gnns.append(nn.Linear(hidden_dim, next_dim))
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.activations.append(get_activation(gnn_activation))
        
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(drop_ratio)
        self.enc_to_dec = torch.nn.Linear(in_dim, hidden_dim, bias=False)

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(gnn_layer-1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(next_dim))
        if zero_mask:
            self.mask_embed = nn.Parameter(torch.zeros((1, hidden_dim,)), requires_grad=False)
        else:
            self.mask_embed = nn.Parameter(torch.zeros((1, hidden_dim,)))
            nn.init.normal_(self.mask_embed, std=.02)

        self.trans_layer = trans_layer
        if self.trans_layer > 0:
            if self.gnn_jk == 'last':
                self.gnn2trans = nn.Linear(hidden_dim+pe_dim, d_model, bias=False)
            elif self.gnn_jk == 'concat':
                self.gnn2trans = nn.Linear(hidden_dim * gnn_layer+pe_dim, d_model, bias=True)
            else:
                raise NotImplementedError()
            self.gnn2trans_act = get_activation(gnn_activation)
            self.trans2out = nn.Linear(d_model, out_dim, bias=False)
            self.trans_enc = TransformerNodeEncoder_v3(d_model, trans_layer, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans=custom_trans)

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            self.input_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr, masked_tokens, batch, pe_tokens=None, remask=True):
        x = self.activation(x)
        x = self.enc_to_dec(x)

        if remask:
            if self.drop_mask_tokens:
                ## recover the masked tokens
                box = self.mask_embed.repeat(batch.shape[0], 1)
                box[~masked_tokens] = x
                x = box
            else:
                ## re-masking
                x = torch.where(masked_tokens.reshape(-1, 1), self.mask_embed, x)
        else:
            pass

        if self.use_input_norm:
            x = self.input_norm(x)

        xs = []
        for layer in range(self.num_layer):
            x = self.gnns[layer](x, edge_index, edge_attr)
            x = self.batch_norms[layer](x)
            if layer != self.num_layer - 1 or self.gnn_jk == 'concat':
                x = self.activations[layer](x)
            x = self.dropout(x)
            xs.append(x)
        
        if self.trans_layer > 0:
            if pe_tokens is not None:
                x = torch.cat((x, pe_tokens), dim=-1)
            if self.gnn_jk == 'concat':
                x = torch.cat(xs, dim=-1)

            x = self.gnn2trans_act(self.gnn2trans(x))
            assert batch is not None
            pad_x, pad_mask = to_dense_batch(x, batch) # shape = [B, N_max, D], shape = [B, N_max]
            pad_x = pad_x.permute(1, 0, 2)
            pad_out, _ = self.trans_enc(pad_x, ~pad_mask) # discard the cls token; shape = [N_max+1, B, D]
            pad_out = pad_out[:-1] # discard the cls token; shape = [N_max, B, D]
            trans_out = pad_out.permute(1, 0, 2)[pad_mask]
            trans_out = self.trans2out(trans_out)
            x = trans_out
        else:
            if self.gnn_jk == 'last':
                x = xs[-1]
            elif self.gnn_jk == 'concat':
                x = self.combine(torch.cat(xs, dim=-1))
            else:
                raise NotImplementedError()
        return x


class NoEdgeDecoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, gnn_layer=1, drop_ratio = 0, gnn_type = "gin", gnn_activation='relu', gnn_jk='last',
    d_model=128, trans_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False, pe_dim=0, use_input_norm=False, zero_mask=False):
        super().__init__()
        self.gnn_jk = gnn_jk
        self.num_layer = gnn_layer
        self.drop_mask_tokens = drop_mask_tokens
        self.gnns = nn.ModuleList()
        self.activations = nn.ModuleList()
        for layer in range(gnn_layer-1):
            if gnn_type == "gin":
                self.gnns.append(ParaGINConv(hidden_dim, hidden_dim, aggr = "add"))
            else:
                raise NotImplementedError(f"{gnn_type}")
            self.activations.append(get_activation(gnn_activation))
        
        if trans_layer > 0:
            next_dim = hidden_dim
        else:
            if gnn_jk == 'concat':
                self.combine = nn.Linear(hidden_dim * gnn_layer, out_dim)
                next_dim = hidden_dim
            elif gnn_jk == 'last':
                next_dim = out_dim
            else:
                raise NotImplementedError()

        if gnn_type == "gin":
            self.gnns.append(ParaGINConv(hidden_dim, next_dim, aggr = "add"))
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.activations.append(get_activation(gnn_activation))
        
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(drop_ratio)
        self.enc_to_dec = torch.nn.Linear(in_dim, hidden_dim, bias=False)

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(gnn_layer-1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(next_dim))
        if zero_mask:
            self.mask_embed = nn.Parameter(torch.zeros((1, hidden_dim,)), requires_grad=False)
        else:
            self.mask_embed = nn.Parameter(torch.zeros((1, hidden_dim,)))
            nn.init.normal_(self.mask_embed, std=.02)

        self.trans_layer = trans_layer
        if self.trans_layer > 0:
            if self.gnn_jk == 'last':
                self.gnn2trans = nn.Linear(hidden_dim+pe_dim, d_model, bias=False)
            elif self.gnn_jk == 'concat':
                self.gnn2trans = nn.Linear(hidden_dim * gnn_layer+pe_dim, d_model, bias=True)
            else:
                raise NotImplementedError()
            self.gnn2trans_act = get_activation(gnn_activation)
            self.trans2out = nn.Linear(d_model, out_dim, bias=False)
            self.trans_enc = TransformerNodeEncoder_v3(d_model, trans_layer, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans=custom_trans)

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            self.input_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr, masked_tokens, batch, pe_tokens=None, remask=True):
        x = self.activation(x)
        x = self.enc_to_dec(x)

        if remask:
            if self.drop_mask_tokens:
                ## recover the masked tokens
                box = self.mask_embed.repeat(batch.shape[0], 1)
                box[~masked_tokens] = x
                x = box
            else:
                ## re-masking
                x = torch.where(masked_tokens.reshape(-1, 1), self.mask_embed, x)
        else:
            pass

        if self.use_input_norm:
            x = self.input_norm(x)

        xs = []
        for layer in range(self.num_layer):
            x = self.gnns[layer](x, edge_index)
            x = self.batch_norms[layer](x)
            if layer != self.num_layer - 1 or self.gnn_jk == 'concat':
                x = self.activations[layer](x)
            x = self.dropout(x)
            xs.append(x)
        
        if self.trans_layer > 0:
            if pe_tokens is not None:
                x = torch.cat((x, pe_tokens), dim=-1)
            if self.gnn_jk == 'concat':
                x = torch.cat(xs, dim=-1)

            x = self.gnn2trans_act(self.gnn2trans(x))
            assert batch is not None
            pad_x, pad_mask = to_dense_batch(x, batch) # shape = [B, N_max, D], shape = [B, N_max]
            pad_x = pad_x.permute(1, 0, 2)
            pad_out, _ = self.trans_enc(pad_x, ~pad_mask) # discard the cls token; shape = [N_max+1, B, D]
            pad_out = pad_out[:-1] # discard the cls token; shape = [N_max, B, D]
            trans_out = pad_out.permute(1, 0, 2)[pad_mask]
            trans_out = self.trans2out(trans_out)
            x = trans_out
        else:
            if self.gnn_jk == 'last':
                x = xs[-1]
            elif self.gnn_jk == 'concat':
                x = self.combine(torch.cat(xs, dim=-1))
            else:
                raise NotImplementedError()
        return x


class GNNDecoderComplete_v2(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, gnn_layer=1, drop_ratio = 0, gnn_type = "gin", gnn_activation='relu', gnn_jk='last',
    d_model=128, trans_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False, pe_dim=0, use_input_norm=False, zero_mask=False):
        super().__init__()
        self.gnn_jk = gnn_jk
        self.num_layer = gnn_layer
        self.drop_mask_tokens = drop_mask_tokens
        self.gnns = nn.ModuleList()
        self.activations = nn.ModuleList()
        for layer in range(gnn_layer-1):
            if gnn_type == 'gin':
                self.gnns.append(GINConvComplete(hidden_dim, hidden_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v2':
                self.gnns.append(GINConvComplete_v2(hidden_dim, hidden_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v3':
                self.gnns.append(GINConvComplete_v3(hidden_dim, hidden_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v3_relu':
                self.gnns.append(GINConvComplete_v3_relu(hidden_dim, hidden_dim, aggr = "add", act_func=gnn_activation))
            elif gnn_type == 'gin_v4':
                self.gnns.append(GINConvComplete_v4(hidden_dim, hidden_dim, aggr = "add", act_func=gnn_activation))
            else:
                raise NotImplementedError(f"{gnn_type}")
            self.activations.append(get_activation(gnn_activation))
        
        if trans_layer > 0:
            next_dim = hidden_dim
        else:
            if gnn_jk == 'concat':
                self.combine = nn.Linear(hidden_dim * gnn_layer, out_dim)
                next_dim = hidden_dim
            elif gnn_jk == 'last':
                next_dim = out_dim
            else:
                raise NotImplementedError()

        if gnn_type == 'gin':
            self.gnns.append(GINConvComplete(hidden_dim, next_dim, aggr = "add", act_func=gnn_activation))
        elif gnn_type == 'gin_v2':
            self.gnns.append(GINConvComplete_v2(hidden_dim, next_dim, aggr = "add", act_func=gnn_activation))
        elif gnn_type == 'gin_v3':
            self.gnns.append(GINConvComplete_v3(hidden_dim, next_dim, aggr = "add", act_func=gnn_activation))
        elif gnn_type == 'gin_v3_relu':
            self.gnns.append(GINConvComplete_v3_relu(hidden_dim, next_dim, aggr = "add", act_func=gnn_activation))
        elif gnn_type == 'gin_v4':
            self.gnns.append(GINConvComplete_v4(hidden_dim, next_dim, aggr = "add", act_func=gnn_activation))
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.activations.append(get_activation(gnn_activation))
        
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(drop_ratio)
        self.enc_to_dec = torch.nn.Linear(in_dim, hidden_dim, bias=False)

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(gnn_layer-1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(next_dim))
        if zero_mask:
            self.mask_embed = nn.Parameter(torch.zeros((1, hidden_dim,)), requires_grad=False)
        else:
            self.mask_embed = nn.Parameter(torch.zeros((1, hidden_dim,)))
            nn.init.normal_(self.mask_embed, std=.02)

        self.trans_layer = trans_layer
        if self.trans_layer > 0:
            if self.gnn_jk == 'last':
                self.gnn2trans = nn.Linear(hidden_dim+pe_dim, d_model, bias=False)
            elif self.gnn_jk == 'concat':
                self.gnn2trans = nn.Linear(hidden_dim * gnn_layer+pe_dim, d_model, bias=True)
            else:
                raise NotImplementedError()
            self.gnn2trans_act = get_activation(gnn_activation)
            self.trans2out = nn.Linear(d_model, out_dim, bias=False)
            self.trans_enc = TransformerNodeEncoder_v3(d_model, trans_layer, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans=custom_trans)

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            self.input_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr, masked_tokens, batch, pe_tokens=None, remask=True):
        x = self.activation(x)
        x = self.enc_to_dec(x)

        if remask:
            if self.drop_mask_tokens:
                ## recover the masked tokens
                box = self.mask_embed.repeat(batch.shape[0], 1)
                box[~masked_tokens] = x
                x = box
            else:
                ## re-masking
                x = torch.where(masked_tokens.reshape(-1, 1), self.mask_embed, x)

        if self.use_input_norm:
            x = self.input_norm(x)

        xs = []
        for layer in range(self.num_layer):
            x = self.gnns[layer](x, edge_index, edge_attr)
            x = self.batch_norms[layer](x)
            if layer != self.num_layer - 1 or self.gnn_jk == 'concat':
                x = self.activations[layer](x)
            x = self.dropout(x)
            xs.append(x)
        
        if self.trans_layer > 0:
            if pe_tokens is not None:
                x = torch.cat((x, pe_tokens), dim=-1)
            if self.gnn_jk == 'concat':
                x = torch.cat(xs, dim=-1)

            x = self.gnn2trans_act(self.gnn2trans(x))
            assert batch is not None
            pad_x, pad_mask = to_dense_batch(x, batch) # shape = [B, N_max, D], shape = [B, N_max]
            pad_x = pad_x.permute(1, 0, 2)
            pad_out, _ = self.trans_enc(pad_x, ~pad_mask) # discard the cls token; shape = [N_max+1, B, D]
            pad_out = pad_out[:-1] # discard the cls token; shape = [N_max, B, D]
            trans_out = pad_out.permute(1, 0, 2)[pad_mask]
            trans_out = self.trans2out(trans_out)
            x = trans_out
        else:
            if self.gnn_jk == 'last':
                x = xs[-1]
            elif self.gnn_jk == 'concat':
                x = self.combine(torch.cat(xs, dim=-1))
            else:
                raise NotImplementedError()
        return x


class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super().__init__()
        self._dec_type = gnn_type 
        if gnn_type == "gin":
            self.conv = GINConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "gcn":
            self.conv = GCNConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "linear":
            self.dec = torch.nn.Linear(hidden_dim, out_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)    
        self.activation = torch.nn.PReLU() 


    def forward(self, x, edge_index, edge_attr, mask_node_indices):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            x = self.activation(x)
            x = self.enc_to_dec(x)
            x[mask_node_indices] = 0
            out = self.conv(x, edge_index, edge_attr)
        return out


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


if __name__ == "__main__":
    pass

