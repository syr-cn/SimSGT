import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
import custom_transformer as custom_nn

if True:
    # in graphcl:
    num_atom_type = 120 #including the extra mask tokens
    num_chirality_tag = 3
else:
    # in graphtrans:
    num_atom_type = 119
    num_chirality_tag = 4

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class MixedBondEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.edge_embedding = nn.Linear(num_bond_type + num_bond_direction, emb_dim, bias=False)
        nn.init.xavier_uniform_(self.edge_embedding.weight[:, :num_bond_type].T)
        nn.init.xavier_uniform_(self.edge_embedding.weight[:, num_bond_type:].T)

    def forward(self, edge_attr):
        if edge_attr.shape[1] == 2:
            embedding = self.edge_embedding.weight.T
            edge_attr = F.embedding(edge_attr[:, 0], embedding[: num_bond_type]) + F.embedding(edge_attr[:, 1], embedding[num_bond_type: ])
            return edge_attr
        else:
            return self.edge_embedding(edge_attr)


class MixedAtomEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.x_embedding = nn.Linear(num_atom_type + num_chirality_tag, emb_dim, bias=False)
        nn.init.xavier_uniform_(self.x_embedding.weight[:, :num_atom_type].T)
        nn.init.xavier_uniform_(self.x_embedding.weight[:, num_atom_type:].T)

    def forward(self, x):
        if x.shape[1] == 2:
            embedding = self.x_embedding.weight.T
            x = F.embedding(x[:, 0], embedding[: num_atom_type]) + F.embedding(x[:, 1], embedding[num_atom_type: ])
            return x
        else:
            return self.x_embedding(x)


class GINConv(MessagePassing):
    def __init__(self, emb_dim: int):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = MixedBondEncoder(emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        edge_embeddings = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embeddings, weight=edge_weight))
        return out

    def message_error(self, x_j, edge_attr, weight=None):
        if weight is not None:
            return F.relu(x_j + weight.view(-1, 1) * edge_attr)
        else:
            return F.relu(x_j + edge_attr)

    def message(self, x_j, edge_attr, weight=None):
        if weight is not None:
            return F.relu(x_j * weight.view(-1, 1) + edge_attr)
        else:
            return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate nodse embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    @staticmethod
    def need_deg():
        return False

    def __init__(self, num_layer, emb_dim, node_encoder, drop_ratio=0.5, JK="last", residual=False, gnn_type="gin"):
        """
        emb_dim (int): node embedding dimensionality
        num_layer (int): number of GNN message passing layers
        """

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        node_depth = batched_data.node_depth if hasattr(batched_data, "node_depth") else None
        
        edge_weight = batched_data.edge_weight if hasattr(batched_data, 'edge_weight') else None
            

        ### computing input node embedding
        if self.node_encoder is not None:
            encoded_node = (
                self.node_encoder(x) if node_depth is None
                else self.node_encoder(x, node_depth.view(-1,),)
            )
        else:
            encoded_node = x
        tmp = encoded_node + perturb if perturb is not None else encoded_node
        h_list = [tmp]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr, edge_weight)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        elif self.JK == "cat":
            node_representation = torch.cat([h_list[0], h_list[-1]], dim=-1)

        return node_representation


class TransformerNodeEncoder_v3(nn.Module):
    def __init__(self, d_model, num_encoder_layers, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans):
        super().__init__()
        self.norm_input = None
        self.custom_trans = custom_trans
        if custom_trans:
            # Creating Transformer Encoder Model
            encoder_layer = custom_nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, transformer_dropout, transformer_activation
            )
            encoder_norm = custom_nn.MaskedBatchNorm1d(d_model)
            self.transformer = custom_nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
            if transformer_norm_input:
                self.norm_input = custom_nn.MaskedBatchNorm1d(d_model)
        else:
            # Creating Transformer Encoder Model
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, transformer_dropout, transformer_activation
            )
            encoder_norm = nn.LayerNorm(d_model)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

            if transformer_norm_input:
                self.norm_input = nn.LayerNorm(d_model)
        self.cls_embedding = None
        
        # if args.self_graph_pooling== "cls": 
        ## add cls by default; following MAE; following MAE initialization
        self.cls_embedding = nn.Parameter(torch.zeros([1, 1, d_model], requires_grad=True))
        nn.init.normal_(self.cls_embedding, std=.02)

    def forward(self, padded_h_node, src_padding_mask):
        """
        padded_h_node: N_max x B x D
        src_padding_mask: B x N_max
        """

        # (S, B, h_d), (B, S)

        if self.cls_embedding is not None:
            expand_cls_embedding = self.cls_embedding.expand(1, padded_h_node.size(1), -1) # shape = [1, B, D]
            padded_h_node = torch.cat([padded_h_node, expand_cls_embedding], dim=0) # shape = [N_max+1, B, D]

            zeros = src_padding_mask.new_zeros(src_padding_mask.size(0), 1) # shape = [B, 1]
            src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1) # shape = [B, N_max+1]
        
        if self.norm_input is not None:
            if self.custom_trans:
                padded_h_node = self.norm_input(padded_h_node, ~src_padding_mask)
            else:
                padded_h_node = self.norm_input(padded_h_node)
        transformer_out = self.transformer(padded_h_node, src_key_padding_mask=src_padding_mask)  # (N_max+1, B, D)
        return transformer_out, src_padding_mask


class TransformerNodeDecoder(nn.Module):
    def __init__(self, d_model, num_encoder_layers, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans):
        super().__init__()
        self.norm_input = None
        self.custom_trans = custom_trans
        if custom_trans:
            # Creating Transformer Encoder Model
            decoder_layer = custom_nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, transformer_dropout, transformer_activation
            )
            decoder_norm = custom_nn.MaskedBatchNorm1d(d_model)
            self.transformer = custom_nn.TransformerDecoder(decoder_layer, num_encoder_layers, decoder_norm)
            if transformer_norm_input:
                self.norm_input = custom_nn.MaskedBatchNorm1d(d_model)
        else:
            # Creating Transformer Encoder Model
            decoder_layer = nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, transformer_dropout, transformer_activation
            )
            decoder_norm = nn.LayerNorm(d_model)
            self.transformer = nn.TransformerDecoder(decoder_layer, num_encoder_layers, decoder_norm)

            if transformer_norm_input:
                self.norm_input = nn.LayerNorm(d_model)

    def forward(self, padded_tgt, padded_memory, tgt_mask, memory_mask):
        """
        padded_tgt: N_max x B x D
        tgt_mask: B x N_max
        """
        # (S, B, h_d), (B, S)
        if self.norm_input is not None:
            if self.custom_trans:
                padded_tgt = self.norm_input(padded_tgt, ~tgt_mask)
            else:
                padded_tgt = self.norm_input(padded_tgt)
        transformer_out = self.transformer(padded_tgt, padded_memory, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=memory_mask)  # (N_max+1, B, D)
        return transformer_out


class TransformerNodeEncoder_v2(nn.Module):
    def __init__(self, d_model, num_encoder_layers, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input):
        super().__init__()
        # Creating Transformer Encoder Model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, transformer_dropout, transformer_activation
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.norm_input = None
        if transformer_norm_input:
            self.norm_input = nn.LayerNorm(d_model)
        self.cls_embedding = None
        
        # if args.self_graph_pooling== "cls": 
        ## add cls by default; following MAE; following MAE initialization
        self.cls_embedding = nn.Parameter(torch.zeros([1, 1, d_model], requires_grad=True))
        nn.init.normal_(self.cls_embedding, std=.02)

    def forward(self, padded_h_node, src_padding_mask):
        """
        padded_h_node: N_max x B x D
        src_padding_mask: B x N_max
        """

        # (S, B, h_d), (B, S)

        if self.cls_embedding is not None:
            expand_cls_embedding = self.cls_embedding.expand(1, padded_h_node.size(1), -1) # shape = [1, B, D]
            padded_h_node = torch.cat([padded_h_node, expand_cls_embedding], dim=0) # shape = [N_max+1, B, D]

            zeros = src_padding_mask.new_zeros(src_padding_mask.size(0), 1) # shape = [B, 1]
            src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1) # shape = [B, N_max+1]
        if self.norm_input is not None:
            padded_h_node = self.norm_input(padded_h_node)

        transformer_out = self.transformer(padded_h_node, src_key_padding_mask=src_padding_mask)  # (N_max+1, B, D)
        return transformer_out, src_padding_mask


class TransformerNodeEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.num_layer = args.num_encoder_layers
        # Creating Transformer Encoder Model
        encoder_layer = nn.TransformerEncoderLayer(
            args.d_model, args.nhead, args.dim_feedforward, args.transformer_dropout, args.transformer_activation
        )
        encoder_norm = nn.LayerNorm(args.d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, args.num_encoder_layers, encoder_norm)
        self.max_input_len = args.max_input_len

        self.norm_input = None
        if args.transformer_norm_input:
            self.norm_input = nn.LayerNorm(args.d_model)
        self.cls_embedding = None
        
        # if args.self_graph_pooling== "cls": 
        ## add cls by default; following MAE; following MAE initialization
        self.cls_embedding = nn.Parameter(torch.zeros([1, 1, args.d_model], requires_grad=True))
        nn.init.normal_(self.cls_embedding, std=.02)

    def forward(self, padded_h_node, src_padding_mask):
        """
        padded_h_node: N_max x B x D
        src_padding_mask: B x N_max
        """

        # (S, B, h_d), (B, S)

        if self.cls_embedding is not None:
            expand_cls_embedding = self.cls_embedding.expand(1, padded_h_node.size(1), -1) # shape = [1, B, D]
            padded_h_node = torch.cat([padded_h_node, expand_cls_embedding], dim=0) # shape = [N_max+1, B, D]

            zeros = src_padding_mask.new_zeros(src_padding_mask.size(0), 1) # shape = [B, 1]
            src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1) # shape = [B, N_max+1]
        if self.norm_input is not None:
            padded_h_node = self.norm_input(padded_h_node)

        transformer_out = self.transformer(padded_h_node, src_key_padding_mask=src_padding_mask)  # (N_max+1, B, D)
        return transformer_out, src_padding_mask


class GraphTrans(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("GNNTransformer - Training Config")
        group.add_argument('--gnn_emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
        group.add_argument('--gnn_virtual_node', action='store_true', default=False)
        group.add_argument('--gnn_dropout', type=float, default=0) # follow the setting of MAE
        group.add_argument('--gnn_num_layer', type=int, default=5,
                            help='number of GNN message passing layers (default: 5)')
        group.add_argument('--gnn_JK', type=str, default='cat')
        group.add_argument('--gnn_residual', action='store_true', default=False)
        group.add_argument('--self_graph_pooling', type=str, default='none')
        group.add_argument('--d_model', type=int, default=128)
        group.add_argument("--nhead", type=int, default=4, help="transformer heads")
        group.add_argument("--dim_feedforward", type=int, default=512, help="transformer feedforward dim")
        group.add_argument("--transformer_dropout", type=float, default=0) # follow the setting of MAE
        group.add_argument("--transformer_activation", type=str, default="relu")
        group.add_argument("--num_encoder_layers", type=int, default=4)
        group.add_argument("--max_input_len", default=1000, help="The max input length of transformer input")
        group.add_argument("--transformer_norm_input", action="store_true", default=True)
        
    def __init__(self, args, gnn=None):
        super(GraphTrans, self).__init__()

        if gnn is None:
            atom_encoder = MixedAtomEncoder(emb_dim=args.gnn_emb_dim)
            self.gnn_node = GNN_node(
                args.gnn_num_layer,
                args.gnn_emb_dim,
                atom_encoder,
                JK=args.gnn_JK,
                drop_ratio=args.gnn_dropout,
                residual=False,
                gnn_type=args.gnn_type,
            )
        else:
            self.gnn_node = gnn
        gnn_emb_dim = 2 * args.gnn_emb_dim if args.gnn_JK == "cat" else args.gnn_emb_dim
        self.gnn2transformer = nn.Linear(gnn_emb_dim, args.d_model)
        self.transformer_encoder = TransformerNodeEncoder(args)
        self.pooling = args.self_graph_pooling
        self.num_encoder_layers = args.num_encoder_layers
        self.emb_dim = args.d_model


    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        
        if self.num_encoder_layers == 0:
            assert self.pooling in {'mean', 'sum'}
            if self.pooling == 'mean':
                h_graph = global_mean_pool(h_node, batched_data.batch)
            elif self.pooling == 'sum':
                h_graph = global_add_pool(h_node, batched_data.batch)
            elif self.pooling == 'none':
                return h_node
            else:
                raise NotImplementedError
            return h_graph # shape = [B, 2 * 300]

        h_node = self.gnn2transformer(h_node)
        padded_h_node, src_padding_mask = to_dense_batch(h_node, batched_data.batch) # shape = [B, N_max, D], shape = [B, N_max]
        padded_h_node = padded_h_node.permute(1, 0, 2) # shape = [N_max, B, D]
        src_padding_mask = ~src_padding_mask # shape = [B, N_max]

        transformer_out = padded_h_node # shape = [N_max, B, D]
        if self.num_encoder_layers > 0:
            transformer_out, _ = self.transformer_encoder(transformer_out, src_padding_mask) # shape = [N_max+1, B, D], 

        if self.pooling in ["last", "cls"]:
            h_graph = transformer_out[-1]
        elif self.pooling == "mean":
            h_graph = transformer_out.sum(0) / src_padding_mask.sum(-1, keepdim=True)
        elif self.pooling == 'none':
            return transformer_out[:-1] # shape = [N_max, B, D]
        else:
            raise NotImplementedError

        return h_graph

    def from_pretrained(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)
        print("Loaded GraphTrans from " + path)



if __name__ == '__main__':
    import argparse
    from loader import MoleculeDataset
    from torch_geometric.data import DataLoader
    from graph_trans_model_old import GraphTrans as GraphTrans_ori
    from splitters import scaffold_split, random_split, random_scaffold_split
    import pandas as pd
    torch.set_default_tensor_type(torch.DoubleTensor)

    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--gnn_type', type=str, default='gin')

    parser.add_argument('--dataset', type=str, default='hiv', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default="models_graphtrans/graphtrans_attn-nce-temp0.1-lr0.0001-epoch60.pth",
                        help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=12344, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=2, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--log_file', type=str, default='result.log')
    GraphTrans.add_args(parser)
    args = parser.parse_args()
    args.device = 'cuda:%d' % args.device

    dataset_list = ["tox21", "hiv", "muv", "bace", "bbbp", "toxcast", "sider", "clintox"]
    for dataset in dataset_list:
        args.dataset = dataset
        # for dataset in []
        if args.dataset == "tox21":
            num_tasks = 12
            epoch_num = 100
        elif args.dataset == "hiv":
            num_tasks = 1
            epoch_num = 100
        elif args.dataset == "pcba":
            num_tasks = 128
            epoch_num = 100
        elif args.dataset == "muv":
            num_tasks = 17
            epoch_num = 100
        elif args.dataset == "bace":
            num_tasks = 1
            epoch_num = 100
        elif args.dataset == "bbbp":
            num_tasks = 1
            epoch_num = 100
        elif args.dataset == "toxcast":
            num_tasks = 617
            epoch_num = 100
        elif args.dataset == "sider":
            num_tasks = 27
            epoch_num = 100
            args.batch_size = min(64, args.batch_size)
        elif args.dataset == "clintox":
            num_tasks = 2
            epoch_num = 300
        else:
            raise ValueError("Invalid dataset name.")
        args.input_model_file = ''
        
        gnn = GraphTrans(args)
        gnn_ori = GraphTrans_ori(args)

        ## some necessary edit so we can load the state dict
        ori_state_dict = gnn_ori.state_dict()
        target_state_dict = gnn.state_dict()
        new_state_dict = {}
        for key in ori_state_dict:
            if key in target_state_dict:
                new_state_dict[key] = ori_state_dict[key]
        new_state_dict['gnn_node.node_encoder.x_embedding.weight'] = torch.cat((ori_state_dict['gnn_node.node_encoder.atom_embedding_list.0.weight'], ori_state_dict['gnn_node.node_encoder.atom_embedding_list.1.weight']), dim=0).T
        for i in range(args.gnn_num_layer):
            new_state_dict['gnn_node.convs.%d.bond_encoder.edge_embedding.weight'%i] = torch.cat((ori_state_dict['gnn_node.convs.%d.edge_embedding1.weight' % i], ori_state_dict['gnn_node.convs.%d.edge_embedding2.weight' % i]), dim=0).T
        
        gnn.load_state_dict(new_state_dict)
        
        gnn.to(args.device)
        gnn_ori.to(args.device)
        
        gnn.eval()
        gnn_ori.eval()
        
        root_dir = '/storage_fast/zyliu/code/E-GCL-Private/transferLearning_MoleculeNet_PPI/chem/dataset/'
        dataset = MoleculeDataset(root_dir + args.dataset, dataset=args.dataset)

        print(args.dataset)
    
        if args.split == "scaffold":
            smiles_list = pd.read_csv(root_dir + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
            print("scaffold")
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
            print("random")
        elif args.split == "random_scaffold":
            smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
            print("random scaffold")
        else:
            raise ValueError("Invalid split option.")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        for step, batch in enumerate(train_loader):
            batch = batch.to(args.device)
            out1 = gnn_ori(batch)
            onehot_x = torch.cat((F.one_hot(batch.x[:, 0], num_atom_type), F.one_hot(batch.x[:, 1], num_chirality_tag)), dim=-1).double()
            onehot_e = torch.cat((F.one_hot(batch.edge_attr[:, 0], num_bond_type), F.one_hot(batch.edge_attr[:, 1], num_bond_direction)), dim=-1).double()
            batch.x = onehot_x
            batch.edge_attr = onehot_e
            out2 = gnn(batch)
            
            closeness = torch.isclose(out1, out2)
            distance = (out1 - out2).abs().sum()
            assert closeness.all(), print(closeness.sum(), closeness.shape[0] * closeness.shape[1], distance)
            if step > 10:
                break
        print(f'Pass test on {args.dataset}')
        