"""
SignNet https://arxiv.org/abs/2202.13013
based on https://github.com/cptq/SignNet-BasisNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing


num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 


class GINEConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, mlp, emb_dim, aggr = "add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINEConv, self).__init__(**kwargs)
        #multi-layer perceptron
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), nn.BatchNorm1d(2*emb_dim), torch.nn.Linear(2*emb_dim, out_dim))
        self.mlp = mlp
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.add_selfloop = False

    def forward(self, x, edge_index, edge_attr):
        '''
        '''
        # #add self loops in the edge space
        if self.add_selfloop:
            edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        '''
        x_j: shape = [K, E, D]
        edge_attr: shape = [E, D]
        '''
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class CustomDropout(nn.Module):
    '''
    This dropout layer will use the same dropout 
    in the consecutive two forward runs
    '''
    def __init__(self, dropout_rate):
        super().__init__()
        assert 0 <= dropout_rate < 1
        self.dropout_rate = dropout_rate
        self.eps = 1e-8

    def forward(self, input_):
        if not self.training:
            return input_
        with torch.no_grad():
            if hasattr(self, 'mask'):
                mask = self.mask
                delattr(self, 'mask')
            else:
                mask = torch.rand_like(input_) > self.dropout_rate
                self.mask = mask
        input_ = input_ / (1-self.dropout_rate + self.eps) * mask
        return input_


class GIN_v2(torch.nn.Module):
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
    def __init__(self, in_dim, h_dim, out_dim, num_layer, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        assert self.num_layer >= 3
        ###List of MLPs
        self.gnns = nn.ModuleList()
        mlp = nn.Sequential(nn.Linear(in_dim, 2*h_dim), nn.LeakyReLU(0.2), nn.Linear(2*h_dim, h_dim))
        self.gnns.append(GINConv(mlp))
        for layer in range(num_layer-2):
            mlp = nn.Sequential(nn.Linear(h_dim, 2*h_dim), nn.LeakyReLU(0.2), nn.Linear(2*h_dim, h_dim))
            self.gnns.append(GINConv(mlp))
        mlp = nn.Sequential(nn.Linear(h_dim, 2*h_dim), nn.LeakyReLU(0.2), nn.Linear(2*h_dim, out_dim))
        self.gnns.append(GINConv(mlp))
        self.activation = nn.LeakyReLU(0.2)
        self.dropouts = nn.ModuleList()
        for layer in range(num_layer):
            self.dropouts.append(CustomDropout(self.drop_ratio))

    def forward(self, x, edge_index):
        '''
        x: shape = [K, N, C]
        '''
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = self.dropouts[layer](h)
            else:
                h = self.dropouts[layer](self.activation(h))
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


class GINE(torch.nn.Module):
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
    def __init__(self, in_dim, h_dim, out_dim, num_layer, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        assert self.num_layer >= 3

        ###List of MLPs
        self.embedding = nn.Linear(in_dim, h_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.embedding.weight.T)
        self.gnns = nn.ModuleList()
        
        for layer in range(num_layer-1):
            mlp = nn.Sequential(nn.Linear(h_dim, 2*h_dim), nn.ReLU(), nn.Linear(2*h_dim, h_dim))
            self.gnns.append(GINEConv(mlp, h_dim))
        
        mlp = nn.Sequential(nn.Linear(h_dim, 2*h_dim), nn.ReLU(), nn.Linear(2*h_dim, out_dim))
        self.gnns.append(GINEConv(mlp, h_dim))
        
        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(h_dim))

    def forward(self, x, edge_index, edge_attr):
        '''
        x: shape = [K, N, C]
        '''
        x = self.embedding(x)
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            if x.ndim == 2:
                h = self.batch_norms[layer](h)
            elif x.ndim == 3:
                h = self.batch_norms[layer](h.transpose(2, 1)).transpose(2, 1)
            else:
                raise ValueError('invalid dimension of x')

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



class MaskedGINDeepSigns_v3(nn.Module):
    """ Sign invariant neural network with sum pooling and DeepSet.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dim_pe, rho_num_layers, dropout=0.0):
        super().__init__()
        self.enc = GIN_v2(in_channels, hidden_channels, out_channels, num_layers, drop_ratio=dropout)
        assert rho_num_layers == 2
        self.rho = nn.Sequential(nn.Linear(out_channels, hidden_channels), nn.LeakyReLU(0.2), nn.Linear(hidden_channels, dim_pe), nn.LeakyReLU(0.2))

    def forward(self, x_pos, x_neg, edge_index, empty_mask):
        x_pos = x_pos.transpose(0, 1)  # N x K x In -> K x N x In
        x_neg = x_neg.transpose(0, 1)  # N x K x In -> K x N x In
        x = self.enc(x_pos, edge_index) + self.enc(x_neg, edge_index)  # K x N x Out
        x = x.transpose(0, 1)  # K x N x Out -> N x K x Out

        x[empty_mask] = 0
        x = x.sum(dim=1)  # (sum over K) -> N x Out
        x = self.rho(x)  # N x Out -> N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x


class SignNetNodeEncoder_v3(torch.nn.Module):
    """SignNet Positional Embedding node encoder.
    https://arxiv.org/abs/2202.13013
    https://github.com/cptq/SignNet-BasisNet

    Uses precomputated Laplacian eigen-decomposition, but instead
    of eigen-vector sign flipping + DeepSet/Transformer, computes the PE as:
    SignNetPE(v_1, ... , v_k) = \rho ( [\phi(v_i) + \rhi(−v_i)]^k_i=1 )
    where \phi is GIN network applied to k first non-trivial eigenvectors, and
    \rho is an MLP if k is a constant, but if all eigenvectors are used then
    \rho is DeepSet with sum-pooling.

    SignNetPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with SignNetPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, args):
        super().__init__()
        dim_pe = args.dim_pe  # Size of PE embedding
        sign_inv_layers = args.layers  # Num. layers in \phi GNN part
        rho_layers = args.post_layers  # Num. layers in \rho MLP/DeepSet
        assert rho_layers >= 1

        self.sign_inv_net = MaskedGINDeepSigns_v3(
            in_channels=2,
            hidden_channels=args.phi_hidden_dim,
            out_channels=args.phi_out_dim,
            num_layers=sign_inv_layers,
            dim_pe=dim_pe,
            rho_num_layers=rho_layers,
            dropout=args.gnn_dropout,
        )

    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}")
        eigvals = batch.EigVals # (Num nodes) x (Num Eigenvectors)
        eigvecs = batch.EigVecs
        pos_enc_pos = torch.stack((eigvals, eigvecs), dim=2)  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc_neg = torch.stack((eigvals, -eigvecs), dim=2)  # (Num nodes) x (Num Eigenvectors) x 2

        empty_mask = torch.isnan(eigvals)
        pos_enc_pos[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc_neg[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2

        # SignNet
        pos_enc = self.sign_inv_net(pos_enc_pos, pos_enc_neg, batch.edge_index, empty_mask)  # (Num nodes) x (pos_enc_dim)

        return pos_enc


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 use_bn=False, use_ln=False, dropout=0.5, activation='relu',
                 residual=False):
        super().__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()

        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual

    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2, 1)).transpose(2, 1)
                else:
                    raise ValueError('invalid dimension of x')
            if self.use_ln: x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers,
                 use_bn=True, dropout=0.5, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        self.use_bn = use_bn
        # input layer
        update_net = MLP(in_channels, hidden_channels, hidden_channels, 2,
                         use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net))
        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(hidden_channels, hidden_channels, hidden_channels,
                             2, use_bn=use_bn, dropout=dropout,
                             activation=activation)
            self.layers.append(GINConv(update_net))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        update_net = MLP(hidden_channels, hidden_channels, out_channels, 2,
                         use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net))
        if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i - 1](x)
                    elif x.ndim == 3:
                        x = self.bns[i - 1](x.transpose(2, 1)).transpose(2, 1)
                    else:
                        raise ValueError('invalid x dim')
            x = layer(x, edge_index)
        return x


class GINDeepSigns(nn.Module):
    """ Sign invariant neural network with MLP aggregation.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 k, dim_pe, rho_num_layers, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        rho_dim = out_channels * k
        self.rho = MLP(rho_dim, hidden_channels, dim_pe, rho_num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        x = x.transpose(0, 1) # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)
        x = x.transpose(0, 1).reshape(N, -1)  # K x N x Out -> N x (K * Out)
        x = self.rho(x)  # N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x


class MaskedGINDeepSigns(nn.Module):
    """ Sign invariant neural network with sum pooling and DeepSet.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dim_pe, rho_num_layers, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        self.rho = MLP(out_channels, hidden_channels, dim_pe, rho_num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)

    def batched_n_nodes(self, batch_index):
        batch_size = batch_index.max().item() + 1
        one = batch_index.new_ones(batch_index.size(0))
        n_nodes = scatter(one, batch_index, dim=0, dim_size=batch_size,
                          reduce='add')  # Number of nodes in each graph.
        n_nodes = n_nodes.unsqueeze(1)
        return torch.cat([size * n_nodes.new_ones(size) for size in n_nodes])

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        K = x.shape[1]  # Max. number of eigen vectors / frequencies.
        x = x.transpose(0, 1)  # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)  # K x N x Out
        x = x.transpose(0, 1)  # K x N x Out -> N x K x Out

        batched_num_nodes = self.batched_n_nodes(batch_index)
        mask = torch.arange(K).unsqueeze(0).expand(N, -1)
        mask = (mask.to(batch_index.device) < batched_num_nodes.unsqueeze(1)).bool()
        # print(f"     - mask: {mask.shape} {mask}")
        # print(f"     - num_nodes: {num_nodes}")
        # print(f"     - batched_num_nodes: {batched_num_nodes.shape} {batched_num_nodes}")
        x[~mask] = 0
        x = x.sum(dim=1)  # (sum over K) -> N x Out
        x = self.rho(x)  # N x Out -> N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x


class SignNetNodeEncoder(torch.nn.Module):
    """SignNet Positional Embedding node encoder.
    https://arxiv.org/abs/2202.13013
    https://github.com/cptq/SignNet-BasisNet

    Uses precomputated Laplacian eigen-decomposition, but instead
    of eigen-vector sign flipping + DeepSet/Transformer, computes the PE as:
    SignNetPE(v_1, ... , v_k) = \rho ( [\phi(v_i) + \rhi(−v_i)]^k_i=1 )
    where \phi is GIN network applied to k first non-trivial eigenvectors, and
    \rho is an MLP if k is a constant, but if all eigenvectors are used then
    \rho is DeepSet with sum-pooling.

    SignNetPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with SignNetPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, args):
        super().__init__()

        # args = cfg.posenc_SignNet
        dim_pe = args.dim_pe  # Size of PE embedding
        model_type = args.model  # Encoder NN model type for SignNet
        if model_type.lower() not in ['mlp', 'deepset']:
            raise ValueError(f"Unexpected SignNet model {model_type}")
        self.model_type = model_type
        self.enable_eigval = args.enable_eigval
        sign_inv_layers = args.layers  # Num. layers in \phi GNN part
        rho_layers = args.post_layers  # Num. layers in \rho MLP/DeepSet
        if rho_layers < 1:
            raise ValueError(f"Num layers in rho model has to be positive.")
        max_freqs = args.max_freqs  # Num. eigenvectors (frequencies)

        in_channels = 2 if self.enable_eigval else 1
        # Sign invariant neural network.
        if self.model_type == 'MLP':
            self.sign_inv_net = GINDeepSigns(
                in_channels=in_channels,
                hidden_channels=args.phi_hidden_dim,
                out_channels=args.phi_out_dim,
                num_layers=sign_inv_layers,
                k=max_freqs,
                dim_pe=dim_pe,
                rho_num_layers=rho_layers,
                use_bn=True,
                dropout=0.0,
                activation='relu'
            )
        elif self.model_type == 'DeepSet':
            self.sign_inv_net = MaskedGINDeepSigns(
                in_channels=in_channels,
                hidden_channels=args.phi_hidden_dim,
                out_channels=args.phi_out_dim,
                num_layers=sign_inv_layers,
                dim_pe=dim_pe,
                rho_num_layers=rho_layers,
                use_bn=True,
                dropout=0.0,
                activation='relu'
            )
        else:
            raise ValueError(f"Unexpected model {self.model_type}")

    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}")
        
        eigvals = batch.EigVals
        eigvecs = batch.EigVecs

        pos_enc = eigvecs.unsqueeze(-1)  # (Num nodes) x (Num Eigenvectors) x 1
        if self.enable_eigval:
            pos_enc = torch.cat((pos_enc, eigvals), dim=2)  # (Num nodes) x (Num Eigenvectors) x 2

        empty_mask = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 1

        # SignNet
        pos_enc = self.sign_inv_net(pos_enc, batch.edge_index, batch.batch)  # (Num nodes) x (pos_enc_dim)

        return pos_enc
    

class MaskedGINDeepSigns_v2(nn.Module):
    """ Sign invariant neural network with sum pooling and DeepSet.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dim_pe, rho_num_layers, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        self.rho = MLP(out_channels, hidden_channels, dim_pe, rho_num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        self.combine = nn.Sequential(nn.Linear(out_channels, out_channels), nn.ReLU())

    def forward(self, x, edge_index, empty_mask):
        x = x.transpose(0, 1)  # N x K x In -> K x N x In
        x = self.combine(self.enc(x, edge_index) + self.enc(-x, edge_index))  # K x N x Out
        x = x.transpose(0, 1)  # K x N x Out -> N x K x Out
        x = torch.where(empty_mask.unsqueeze(-1), 0, x)
        x = x.sum(dim=1)  # (sum over K) -> N x Out
        x = self.rho(x)  # N x Out -> N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x


class SignNetNodeEncoder_v2(torch.nn.Module):
    """SignNet Positional Embedding node encoder.
    https://arxiv.org/abs/2202.13013
    https://github.com/cptq/SignNet-BasisNet

    Uses precomputated Laplacian eigen-decomposition, but instead
    of eigen-vector sign flipping + DeepSet/Transformer, computes the PE as:
    SignNetPE(v_1, ... , v_k) = \rho ( [\phi(v_i) + \rhi(−v_i)]^k_i=1 )
    where \phi is GIN network applied to k first non-trivial eigenvectors, and
    \rho is an MLP if k is a constant, but if all eigenvectors are used then
    \rho is DeepSet with sum-pooling.

    SignNetPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with SignNetPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, args):
        super().__init__()
        dim_pe = args.dim_pe  # Size of PE embedding
        assert args.model.lower() == 'deepset'
        self.enable_eigval = args.enable_eigval
        assert args.post_layers >= 1
        in_channels = 2 if self.enable_eigval else 1
        
        # Sign invariant neural network.        
        self.sign_inv_net = MaskedGINDeepSigns_v2(
            in_channels=in_channels,
            hidden_channels=args.phi_hidden_dim,
            out_channels=args.phi_out_dim,
            num_layers=args.layers,
            dim_pe=dim_pe,
            rho_num_layers=args.post_layers,
            use_bn=True,
            dropout=0.0,
            activation='relu'
        )


    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}")
        eigvals = batch.EigVals
        eigvecs = batch.EigVecs
        pos_enc = torch.cat((eigvecs.unsqueeze(-1), eigvals), dim=2)  # (Num nodes) x (Num Eigenvectors) x 2

        empty_mask = torch.isnan(eigvals.squeeze(-1)) # shape = [num_nodes, num_eigenvectors]
        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 1

        # SignNet
        pos_enc = self.sign_inv_net(pos_enc, batch.edge_index, empty_mask)  # (Num nodes) x (pos_enc_dim)
        return pos_enc

def get_diagonal(P, eye, N):
    out = (P * eye).coalesce()
    indices = out._indices()
    values = out._values()
    diagonal = torch.zeros(N, device=P.device)
    diagonal[indices[0]] = values
    return diagonal


class RWSENodeEncoder(torch.nn.Module):
    """SignNet Positional Embedding node encoder.
    https://arxiv.org/abs/2202.13013
    https://github.com/cptq/SignNet-BasisNet

    Uses precomputated Laplacian eigen-decomposition, but instead
    of eigen-vector sign flipping + DeepSet/Transformer, computes the PE as:
    SignNetPE(v_1, ... , v_k) = \rho ( [\phi(v_i) + \rhi(−v_i)]^k_i=1 )
    where \phi is GIN network applied to k first non-trivial eigenvectors, and
    \rho is an MLP if k is a constant, but if all eigenvectors are used then
    \rho is DeepSet with sum-pooling.

    SignNetPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with SignNetPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, args):
        super().__init__()
        self.dim_pe = args.dim_pe  # Size of PE embedding
        self.layers = args.layers
        self.linear = nn.Linear(args.layers, args.dim_pe)
        self.bn = nn.BatchNorm1d(args.layers)

    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}")
        N = batch.batch.shape[0]
        edge_index = batch.edge_index
        with torch.no_grad():
            nsl = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, nsl]

            ones = torch.ones(edge_index.shape[1], device=batch.batch.device) # shape = [E]
            adj = torch.sparse.FloatTensor(edge_index, ones, size=(N, N)) # shape = [N, N]
            degree = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1) # shape = [N]
            inv_degree = 1 / degree
            P = torch.sparse.FloatTensor(edge_index, inv_degree[edge_index[0]], size=(N, N)) # shape = [N, N]

            eye = torch.ones_like(batch.batch)
            index = torch.arange(N, device=batch.batch.device).reshape(1, -1).expand(2, -1)
            eye = torch.sparse.FloatTensor(index, eye, size=(N, N))
            
            rwse = []
            Pk = P.clone()
            for i in range(self.layers):
                rwse.append(get_diagonal(Pk, eye, N)) # shape = [N]
                Pk = Pk @ P
            rwse = torch.stack(rwse, dim=-1) # shape = [N, layers]
        rwse = self.bn(rwse)
        rwse = self.linear(rwse)
        return rwse


class LapPENodeEncoder(torch.nn.Module):
    """Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, args):
        super().__init__()

        dim_pe = args.dim_pe  # Size of Laplace PE embedding
        model_type = args.model  # Encoder NN model type for PEs
        if model_type not in ['DeepSet']:
            raise ValueError(f"Unexpected PE model {model_type}")
        self.model_type = model_type
        self.enable_eigval = args.enable_eigval
        n_layers = args.layers  # Num. layers in PE encoder model
        post_n_layers = args.post_layers  # Num. layers to apply after pooling
        max_freqs = args.max_freqs  # Num. eigenvectors (frequencies)
        norm_type = args.raw_norm_type.lower()  # Raw PE normalization layer type


        # Initial projection of eigenvalue and the node's eigenvector value
        in_channels = 2 if self.enable_eigval else 1
        self.linear_A = nn.Linear(in_channels, dim_pe)
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        if model_type == 'DeepSet':
            # DeepSet model for LapPE
            layers = []
            if n_layers == 1:
                layers.append(nn.ReLU())
            else:
                self.linear_A = nn.Linear(2, 2 * dim_pe)
                layers.append(nn.ReLU())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(nn.ReLU())
            self.pe_encoder = nn.Sequential(*layers)

        self.post_mlp = None
        if post_n_layers > 0:
            # MLP to apply post pooling
            layers = []
            if post_n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(nn.ReLU())
                for _ in range(post_n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(nn.ReLU())
            self.post_mlp = nn.Sequential(*layers)


    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_LapPE.enable' to True")
        EigVals = batch.EigVals
        EigVecs = batch.EigVecs

        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = EigVecs.unsqueeze(-1)  # (Num nodes) x (Num Eigenvectors) x 1
        if self.enable_eigval:
            pos_enc = torch.cat((pos_enc, EigVals), dim=2)  # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x in_channels

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x in_channels
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe
        pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),
                                               0.)

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe
        
        return pos_enc

class LapPENodeEncoder_v2(torch.nn.Module):
    """Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, args):
        super().__init__()
        print("using lap v2")
        dim_pe = args.dim_pe  # Size of Laplace PE embedding
        model_type = args.model  # Encoder NN model type for PEs
        assert model_type == 'DeepSet'
        self.model_type = model_type
        self.enable_eigval = args.enable_eigval
        n_layers = args.layers  # Num. layers in PE encoder model
        post_n_layers = args.post_layers  # Num. layers to apply after pooling

        # Initial projection of eigenvalue and the node's eigenvector value
        in_channels = 2 if self.enable_eigval else 1

        # DeepSet model for LapPE
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(in_channels, dim_pe))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(in_channels, 2 * dim_pe))
            layers.append(nn.ReLU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(2 * dim_pe, dim_pe))
            layers.append(nn.ReLU())
        self.pe_encoder = nn.Sequential(*layers)

        self.post_mlp = None
        if post_n_layers > 0:
            # MLP to apply post pooling
            layers = []
            if post_n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(nn.ReLU())
                for _ in range(post_n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(nn.ReLU())
            self.post_mlp = nn.Sequential(*layers)

        ## combine layers
        self.combine = nn.Sequential(
            nn.Linear(dim_pe, dim_pe),
            nn.ReLU()
        )


    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_LapPE.enable' to True")
        EigVals = batch.EigVals # shape = [N, max_freq, 1]
        EigVecs = batch.EigVecs.unsqueeze(-1) # shape = [N, max_freq, 1]

        empty_mask = torch.isnan(EigVals.squeeze(-1))  # (Num nodes) x (Num Eigenvectors)
        
        pos_enc0 = torch.cat((EigVecs, EigVals), dim=2)  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc1 = torch.cat((-EigVecs, EigVals), dim=2)  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc0[empty_mask] = 0 # (Num nodes) x (Num Eigenvectors) x in_channels
        pos_enc1[empty_mask] = 0 # (Num nodes) x (Num Eigenvectors) x in_channels
        
        ## encoding
        pos_enc0 = self.pe_encoder(pos_enc0) # (Num nodes) x (Num Eigenvectors) x dim_pe
        pos_enc1 = self.pe_encoder(pos_enc1) # (Num nodes) x (Num Eigenvectors) x dim_pe
        
        pos_enc = self.combine(pos_enc0 + pos_enc1)
        pos_enc = torch.where(empty_mask.unsqueeze(-1), 0, pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe
        
        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False) #/ (~empty_mask).sum(-1, keepdim=True)  # (Num nodes) x dim_pe
        pos_enc = self.post_mlp(pos_enc)
        return pos_enc



class KernelPENodeEncoder(torch.nn.Module):
    """Configurable kernel-based Positional Encoding node encoder.

    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.

    PE of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = 'RWSE'  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, args):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_pe = args.dim_pe  # Size of the kernel-based PE embedding
        kernel_times = args.kernel_times
        if args.kernel_times_func != 'none':
            kernel_times = list(range(*map(int, args.kernel_times_func.split('~'))))
        num_rw_steps = len(kernel_times)
        model_type = args.model.lower()  # Encoder NN model type for PEs
        n_layers = args.layers  # Num. layers in PE encoder model
        norm_type = args.raw_norm_type.lower()  # Raw PE normalization layer type


        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        activation = nn.ReLU()  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, dim_pe))
                layers.append(activation)
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * dim_pe))
                layers.append(activation)
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation)
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation)
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'posenc.kernel.times' values")

        pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        return pos_enc


class PosEncoder(torch.nn.Module):
    """
        Node positional encoding generator
    """
    def __init__(self, args):
        super().__init__()
        self.pe_type = args.pe_type
        if self.pe_type =='signnet':
            self.encoder = SignNetNodeEncoder(args)
        elif self.pe_type == 'signnet_v2':
            self.encoder = SignNetNodeEncoder_v2(args)
        elif self.pe_type == 'signnet_v3':
            self.encoder = SignNetNodeEncoder_v3(args)
        elif self.pe_type =='lap':
            self.encoder = LapPENodeEncoder(args)
        elif self.pe_type =='lap_v2':
            self.encoder = LapPENodeEncoder_v2(args)
        elif self.pe_type =='rwse':
            self.encoder = KernelPENodeEncoder(args)
        elif self.pe_type == 'none':
            pass
        else:
            raise NotImplementedError(f'pe type {self.pe_type} not implemented yet!')
        self.pe_dim = {
            'none':0,
            'lap':args.dim_pe,
            'lap_v2':args.dim_pe,
            'signnet':args.dim_pe,
            'signnet_v2':args.dim_pe,
            'signnet_v3':args.dim_pe,
            'rwse':args.dim_pe
        }[args.pe_type]
    
    def forward(self, batch):
        if self.pe_type =='none':
            pos_enc = None
        else:
            pos_enc = self.encoder(batch)
        return pos_enc
