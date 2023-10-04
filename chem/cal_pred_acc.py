import argparse
from loader import MoleculeDataset
from dataloader import DataLoaderSL
import pickle
import torch
import os
import torch.optim as optim
import utils
from tqdm import tqdm
from rdkit import Chem
import numpy as np
from torch_geometric.loader.dataloader import Collater
import random
from torch_geometric.utils import add_self_loops
import torch.nn as nn
import torch.nn.functional as F
from model import GNN_v2, GNNDecoder_v2, get_activation, PosEncoder, Tokenizer

num_atom_type = 120


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
        group.add_argument('--gnn_token_layer', type=int, default=1)
        group.add_argument('--gnn_encoder_layer', type=int, default=4)
        group.add_argument('--trans_encoder_layer', type=int, default=0)

        ## decoder parameters
        group.add_argument('--gnn_decoder_layer', type=int, default=3)
        group.add_argument('--decoder_input_norm', action='store_true', default=False)
        group.add_argument('--trans_decoder_layer', type=int, default=0)
        
        ## others
        group.add_argument('--nonpara_tokenizer', action='store_true', default=False)
        group.add_argument('--moving_average_decay', type=float, default=0.99)
        group.add_argument('--loss', type=str, default='mse')
        group.add_argument('--loss_all_nodes', action='store_true', default=False)
        group.add_argument('--subgraph_mask', action='store_true', default=False)
        group.add_argument('--zero_mask', action='store_true', default=False)
        group.add_argument('--eps', type=float, default=1)

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
        
        self.tokenizer = GNN_v2(1, gnn_emb_dim, True, JK=gnn_JK, drop_ratio=gnn_dropout, gnn_type=gnn_type, gnn_activation=args.gnn_activation)
        self.gnn_act = get_activation(args.gnn_activation)
        self.encoder = GNN_v2(gnn_encoder_layer, gnn_emb_dim, False, JK=gnn_JK, drop_ratio=gnn_dropout, gnn_type=gnn_type, gnn_activation=args.gnn_activation, 
        d_model=d_model, trans_layer=trans_encoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens, pe_dim=self.pos_encoder.pe_dim)
        self.nonpara_tokenizer = nonpara_tokenizer

        self.mask_embed = nn.Parameter(torch.zeros(gnn_emb_dim))
        nn.init.normal_(self.mask_embed, std=.02)

        if self.nonpara_tokenizer:
            self.tokenizer_nonpara = Tokenizer(gnn_emb_dim, gnn_token_layer, args.eps, JK=gnn_JK, gnn_type='gin')
            
        if gnn_token_layer == 0:
            out_dim = num_atom_type
        else:
            out_dim = gnn_emb_dim

        if trans_encoder_layer > 0:
            in_dim = d_model
        else:
            in_dim = gnn_emb_dim
        
        self.use_trans_decoder = use_trans_decoder
        if self.use_trans_decoder:
            in_dim = d_model + self.pos_encoder.pe_dim
            self.decoder = TransDecoder(in_dim, out_dim, d_model=d_model, trans_layer=trans_decoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens)
            # self.decoder = GNNDecoder_v3(in_dim, gnn_emb_dim, gnn_emb_dim, gnn_decoder_layer, gnn_type=gnn_type, 
            # d_model=d_model, trans_layer=trans_decoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens)
        else:
            print(gnn_decoder_layer, trans_decoder_layer)
            self.decoder = GNNDecoder_v2(in_dim, gnn_emb_dim, out_dim, gnn_decoder_layer, gnn_type=gnn_type, gnn_activation=args.gnn_activation, gnn_jk=args.decoder_jk,
            d_model=d_model, trans_layer=trans_decoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens and trans_encoder_layer > 0, pe_dim=self.pos_encoder.pe_dim, use_input_norm=args.decoder_input_norm, zero_mask=args.zero_mask)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        ## forward tokenizer
        h = self.tokenizer(data.x_masked, edge_index, edge_attr)
        
        # ## forward tokenizer target
        # with torch.no_grad():
        #     if self.nonpara_tokenizer:
        #         g_tokens = self.tokenizer_nonpara(x, edge_index, self.tokenizer.x_embedding1).detach()
        #     else:
        #         g_tokens = self.tokenizer(x, edge_index, edge_attr).detach()

        pe_tokens = self.pos_encoder(data)

        # forward encoder
        h = self.encoder(self.gnn_act(h), edge_index, edge_attr, data.batch, data.mask_tokens, pe_tokens)

        ## forward decoder
        if self.use_trans_decoder:
            g_pred = self.decoder(h, pe_tokens, data.mask_tokens, data.batch)
        else:
            g_pred = self.decoder(h, edge_index, edge_attr, data.mask_tokens, data.batch, pe_tokens)

        ## compute loss
        # if not self.loss_all_nodes:
        #     g_pred = g_pred[data.mask_tokens]
            # g_tokens = g_tokens[data.mask_tokens]

        return g_pred

    def mse_loss(self, x, y):
        loss = ((x - y) ** 2).mean()
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

class DataLoaderSL(torch.utils.data.DataLoader):
    def __init__(
        self,
        mask_ratio, 
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch = None,
        exclude_keys = None,
        **kwargs,
    ):

        self.mask_ratio = mask_ratio 
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs,
        )
        self._collater = Collater(follow_batch, exclude_keys)

    def collate_fn(self, graphs):
        batch = self._collater(graphs)
        if self.mask_ratio > 0:
            ## generate mask idx
            ptr = batch.ptr.tolist()
            mask_idx = []
            for i in range(len(ptr)-1):
                size = ptr[i+1]- ptr[i]
                if size <= 1:
                    continue
                idx = random.sample(range(ptr[i], ptr[i+1]), k=int(size * self.mask_ratio))
                mask_idx.extend(idx)
            mask_tokens = torch.zeros_like(batch.batch, dtype=torch.bool)
            mask_tokens[torch.LongTensor(mask_idx)] = True
            batch.mask_tokens = mask_tokens
            x_masked = batch.x.clone()
            x_masked[mask_tokens] = torch.LongTensor([num_atom_type-1, 0])
            batch.x_masked = x_masked


        ## add self loop
        #add self loops in the edge space
        edge_index, _ = add_self_loops(batch.edge_index, num_nodes = batch.x.size(0))
        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(batch.x.size(0), 2, dtype=torch.long)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        edge_attr = torch.cat((batch.edge_attr, self_loop_attr), dim = 0)
        
    
        batch.edge_index_nosl = batch.edge_index
        batch.edge_attr_nosl = batch.edge_attr
        batch.edge_index = edge_index
        batch.edge_attr = edge_attr
        return batch


def load_subgraphs():
    onehop_path = './figures/zinc_standard_agent_egosub.pkl'
    ### plot subgraph
    with open(onehop_path, 'rb') as f:
        subgraph_counter = pickle.load(f)
    id2subgraph = {idx: k for idx, k in enumerate(subgraph_counter.keys())}
    subgraph2id = {k: idx for idx, k in id2subgraph.items()}
    sub_id2atom_id = {idx: k[0] for idx, k in id2subgraph.items()}
    return id2subgraph, subgraph2id, subgraph_counter, sub_id2atom_id

@torch.no_grad()
def preproces(args, node_embed, id2subgraph, subgraph_counter, norm='l2'):
    # def get_atom_mapping():
    #     mol = Chem.GetPeriodicTable()
    #     mapping = {}
    #     for i in range(1, num_atom_type-1):
    #         mapping[i] = mol.GetElementSymbol(i)
    #     return mapping
    
    # def group_subgraphs_by_center(id2subgraph):
    #     groups = {}
    #     center2gid = {}
    #     id2gid = []
    #     for idx, subgraph in id2subgraph.items():
    #         center, sub = subgraph
    #         if center not in groups:
    #             groups[center] = [idx]
    #             center2gid[center] = len(center2gid)
    #         else:
    #             groups[center].append(idx)
    #         id2gid.append(center2gid[center])
    #     return groups, id2gid, center2gid

    ## encode subgraph
    def subgraph_encoding(subgraph, embedding, eps = 1):
        center, neighbor = subgraph
        neighbor = torch.FloatTensor(neighbor).view(1, -1).to(args.device) # shape = [1, num_atoms, ]
        embedding = neighbor @ embedding + embedding[center].view(1, -1) * (1 + eps) # shape = [1, D]
        return embedding

    # id2sym = get_atom_mapping()
    # sym2id = {v:k for k,v in id2sym.items()}

    # groups, id2gid, center2gid = group_subgraphs_by_center(id2subgraph)
    # id2gid = np.asarray(id2gid)

    emb_bank = []
    counts = []
    for i in range(len(id2subgraph)):
        subgraph = id2subgraph[i]
        counts.append(subgraph_counter[subgraph])
        emb = subgraph_encoding(subgraph, node_embed, eps=0.5) 
        emb_bank.append(emb)
    emb_bank = torch.cat(emb_bank, dim=0) # shape = [N, D]
    counts = torch.FloatTensor(counts).to(args.device).reshape(-1, 1) # shape = [N, 1]
    
    if norm == 'l2':
        emb_bank = F.normalize(emb_bank, dim=-1, p=2)
    elif norm == 'bn':
        emb_bank = nn.BatchNorm1d(300).to(args.device)(emb_bank).detach()
    elif norm == 'mybn':
        # emb_bank; shape = [N, D]
        eps = 1e-8
        mean = emb_bank.mean(dim=0, keepdim=True)
        var = emb_bank.var(dim=0, keepdim=True)
        emb_bank = (emb_bank - mean) / torch.sqrt(var + eps)
    elif norm == 'correct_bn':
        eps = 1e-8
        # counts = torch.ones_like(counts)
        total = counts.sum()
        mean = (emb_bank * counts).sum(dim=0, keepdim=True) / total # shape = [1, D]
        var = ((emb_bank - mean) ** 2 * counts).sum(dim=0, keepdim=True) / total# shape = (1, D)
        emb_bank = (emb_bank - mean) / torch.sqrt(var + eps)
    else:
        raise NotImplementedError()
    return emb_bank

    


def train_mae(args, model, loader, optimizer, epoch):
    loss_meter = utils.AverageMeter()
    model.train()
    train_bar = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(train_bar):
        optimizer.zero_grad()
        batch = batch.to(args.device)
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        loss = float(loss)
        loss_meter.update(loss, batch.num_graphs)
        train_bar.set_description('Epoch:[{:d}/{:d}][{:d}k/{:d}k] AvgLs:{:.4f}; Ls:{:.4f}'.format(epoch, args.epochs, loss_meter.count//1000, len(loader.dataset)//1000, loss_meter.avg, loss))
        if step % args.log_steps == 0:
            utils.write_log('Epoch:[{:d}/{:d}][{:d}k/{:d}k] AvgLs:{:.4f}; Ls:{:.4f}'.format(epoch, args.epochs, loss_meter.count//1000, len(loader.dataset)//1000, loss_meter.avg, loss), log_file=args.log_file, print_=False)
    
    utils.write_log('Epoch:[{:d}/{:d}][{:d}k/{:d}k] AvgLs:{:.4f}; Ls:{:.4f}'.format(epoch, args.epochs, loss_meter.count//1000, len(loader.dataset)//1000, loss_meter.avg, loss), log_file=args.log_file, print_=False)
    
    return loss_meter.avg


def get_subgraph_id(batch, subgraph2id, args):
    x = batch.x[:, 0]
    one_hot_x = F.one_hot(x, num_classes=num_atom_type-1).float() # shape = [N, D]
    ones = torch.ones(batch.edge_index_nosl.shape[1], device=args.device, dtype=torch.float) # shape = [E]
    adj = torch.sparse.FloatTensor(batch.edge_index_nosl, ones, size=(x.shape[0], x.shape[0]))
    subgraphs = torch.sparse.mm(adj, one_hot_x) # shape = [N, D]
    subgraphs = subgraphs.long().tolist() 
    sub_id_list = []
    for i, sub in enumerate(subgraphs):
        ego_sub = (int(x[i]), tuple(sub))
        sub_id = subgraph2id[ego_sub]
        sub_id_list.append(sub_id)
    batch.sub_ids = torch.LongTensor(sub_id_list).to(args.device)
    return batch

@torch.no_grad()
def cal_acc(pred, sub_ids, sub_emb, atom_ids, sub_ids2atom_ids, norm='l2'):
    '''
    pred: shape = [N, D]
    sub_emb: shape = [B, D]
    atom_ids: shape = [N, D]
    '''
    if norm == 'l2':
        pred = F.normalize(pred, p=2, dim=-1)
        sub_emb = F.normalize(sub_emb, p=2, dim=-1)
    
    ### calculate subgraph prediction accuracy
    sim = - torch.cdist(pred, sub_emb) 
    pre_sub_ids = torch.argmax(sim, dim=-1) # shape = [N]
    
    acc = (sub_ids == pre_sub_ids).float().mean()
    acc = float(acc)

    ## calculate atom prediction accuracy
    pre_sub_ids = pre_sub_ids.tolist()
    pre_atom_ids = [sub_ids2atom_ids[i] for i in pre_sub_ids]
    pre_atom_ids = torch.LongTensor(pre_atom_ids)
    atom_acc = (pre_atom_ids == atom_ids.cpu()).float().mean()
    atom_acc = float(atom_acc)
    return acc, atom_acc


@torch.no_grad()
def cal_acc_atom(pred, atom_ids):
    '''
    pred: shape = [N, D]
    sub_emb: shape = [B, D]
    atom_ids: shape = [N, D]
    '''
    pre_id = torch.argmax(pred, dim=-1) # shape = [N,]
    atom_acc = (pre_id == atom_ids.cpu()).float().mean()
    atom_acc = float(atom_acc)
    return atom_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.55,
                        help='dropout ratio (default: 0.55)')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default=None)
    parser.add_argument("--name", type=str, help='Name for log dir')
    parser.add_argument("--save_epochs", type=int, default=20)
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--tlr_scale', type=float, default=1.0)
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--optim_file', type=str, default=None)
    parser.add_argument('--resume_epoch', type=int, default=0)
    TokenMAE.add_args(parser)
    args = parser.parse_args()
    print(args)

    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    utils.set_seed(0)
    log_dir = './results/{}'.format(args.name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_file = os.path.join(log_dir, 'log.txt')
    utils.write_log(str(args), log_file=args.log_file, print_=True)
    
    
    # set up dataset and transform function.
    # if args.pe_type != 'none':
    #     dataset = MoleculeDataset_Eig_v2('dataset_eig/' + dataset_name, dataset=dataset_name, args=args)
    #     dataset = dataset_precomputing(args, dataset)
    # else:
    #     dataset = MoleculeDataset("dataset/" + dataset_name, dataset=dataset_name)

    # loader = DataLoaderSL(args.mask_rate, dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, pin_memory=True)
    # model = TokenMAE(args.encoder_layer, args.token_layer, args.decoder_layer, args.emb_dim, args.nonpara_tokenizer, gnn_JK = args.JK, gnn_drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type,)
    model = TokenMAE(args.gnn_encoder_layer, args.gnn_token_layer, args.gnn_decoder_layer, args.gnn_emb_dim, args.nonpara_tokenizer, args.gnn_JK, args.gnn_dropout, args.gnn_type,
    args.d_model, args.trans_encoder_layer, args.trans_decoder_layer, args.nhead, args.dim_feedforward, args.transformer_dropout, args.transformer_activation, args.transformer_norm_input, custom_trans=args.custom_trans, drop_mask_tokens=args.drop_mask_tokens, use_trans_decoder=args.use_trans_decoder, pe_type=args.pe_type, args=args)

    model = model.to(args.device)
    model.eval()
    # set up optimizers
    params = [
        {'params': model.tokenizer.parameters(), 'lr': args.lr * args.tlr_scale},
        {'params': model.encoder.parameters(), 'lr': args.lr},
        {'params': model.decoder.parameters(), 'lr': args.lr},
        ]
    from pathlib import Path
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.decay)

    
    # path1 = Path('./results/ginv3_pretraining_e4d1_m35')
    # path2 = Path('./results/save_models_ginv3_eps05_pretraining_e4d1_m35')
    path = Path('./results/save_models_ginv3_notk_pretraining_e4d1_m35')
    # model_files = list(path.glob("model_*.pth"))
    model_files = [path / f'model_{i}.pth' for i in range(1, 101, 1)]
    log_path = path / 'log.txt'

    utils.set_seed(1001)
    ## later can try use downstream dataset for prediction
    dataset_name = args.dataset
    dataset = MoleculeDataset("dataset/" + dataset_name, dataset=dataset_name)
    utils.set_seed(1001)
    train_size = int(0.9 * len(dataset))
    test_set = torch.randperm(len(dataset))[train_size:]
    dataset = dataset[test_set]
    loader = DataLoaderSL(args.mask_rate, dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, pin_memory=True)
    
    id2subgraph, subgraph2id, subgraph_counter, sub_id2atom_id = load_subgraphs()
    # sub_emb = preproces(args, model.tokenizer.x_embedding1.weight, id2subgraph, subgraph2id, 'l2')
    batch_list = []
    
    for i, b in tqdm(enumerate(loader)):
        # if i > 3:
        #     break
        b = b.to(args.device)
        b = get_subgraph_id(b, subgraph2id, args)
        batch_list.append(b)

    for file in tqdm(model_files):
        print(file)
        state_dict = torch.load(file, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        sub_emb = preproces(args, model.tokenizer.x_embedding1.weight[:-1], id2subgraph, subgraph_counter, 'correct_bn')
        
        pred_list = []
        sub_id_list = []
        atom_id_list = []
        with torch.no_grad():
            for batch in batch_list:
                pred = model(batch)
                pred_list.append(pred[batch.mask_tokens])
                sub_id_list.append(batch.sub_ids[batch.mask_tokens])
                atom_id_list.append(batch.x[batch.mask_tokens, 0])

        pred_list = torch.cat(pred_list)
        sub_id_list = torch.cat(sub_id_list)
        atom_id_list = torch.cat(atom_id_list)
        if False:
            acc, atom_acc = cal_acc(pred_list, sub_id_list, sub_emb, atom_id_list, sub_id2atom_id, 'none')
        else:
            atom_acc = cal_acc_atom(pred_list.cpu(), atom_id_list)
            acc = 0 
        info = f'model_file {file}; acc: {round(acc * 100, 2)}; atom_acc: {round(atom_acc * 100, 2)}'
        utils.write_log(info, log_path)


    if args.model_file is not None and args.resume_epoch > 0:
        print(f'Loading model from {args.model_file}')
        state_dict = torch.load(args.model_file, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f'Loading optim from {args.optim_file}')
        state_dict = torch.load(args.optim_file, map_location=torch.device('cpu'))
        optimizer.load_state_dict(state_dict)


    

if __name__ == "__main__":
    main()
