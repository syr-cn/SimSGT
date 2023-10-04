import argparse

from loader import MoleculeDataset
from dataloader import DataLoaderSL

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN_v2, GNN_graphpred, TokenMAE, get_activation
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd
from pos_enc.encoder import PosEncoder
import utils

criterion = nn.BCEWithLogitsLoss(reduction = "none")


class TokenMAEClf(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("GNNTransformer - Training Config")
        ## gnn parameters
        group.add_argument('--gnn_emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
        group.add_argument('--gnn_dropout', type=float, default=0.5)
        group.add_argument('--gnn_JK', type=str, default='last')
        group.add_argument('--gnn_type', type=str, default='gin')
        group.add_argument("--gnn_activation", type=str, default="relu")

        ## transformer parameters
        group.add_argument('--d_model', type=int, default=128)
        group.add_argument("--dim_feedforward", type=int, default=512, help="transformer feedforward dim")
        group.add_argument("--nhead", type=int, default=4, help="transformer heads")
        group.add_argument("--transformer_dropout", type=float, default=0.3)
        group.add_argument("--transformer_activation", type=str, default="relu")
        group.add_argument("--transformer_norm_input", action="store_true", default=True)
        group.add_argument('--custom_trans', action='store_true', default=True)
        # group.add_argument("--max_input_len", default=1000, help="The max input length of transformer input")
        
        ## encoder parameters
        group.add_argument('--gnn_token_layer', type=int, default=1)
        group.add_argument('--gnn_encoder_layer', type=int, default=5)
        group.add_argument('--trans_encoder_layer', type=int, default=0)
        group.add_argument('--freeze_token', action='store_true', default=False)
        group.add_argument('--trans_pooling', type=str, default='none')

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

    def __init__(self, freeze_token, token_layer, encoder_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin", 
    d_model=128, trans_encoder_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, args=None):
        super().__init__()
        assert JK == 'last'
        self.freeze_token = freeze_token
        self.tokenizer = GNN_v2(1, emb_dim, True, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type, gnn_activation=args.gnn_activation)
        self.pos_encoder = PosEncoder(args)
        self.gnn_act = get_activation(args.gnn_activation)
        self.encoder = GNN_v2(encoder_layer-1, emb_dim, False, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type, gnn_activation=args.gnn_activation, 
        d_model=d_model, trans_layer=trans_encoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, pe_dim=self.pos_encoder.pe_dim, trans_pooling=args.trans_pooling)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.freeze_token:
            with torch.no_grad():
                h = self.tokenizer(x, edge_index, edge_attr).detach()
        else:
            h = self.tokenizer(x, edge_index, edge_attr)
        
        h = self.encoder(self.gnn_act(h), edge_index, edge_attr, data.batch, pe_tokens=None)
        return h


def train(args, model, device, loader, optimizer):
    loss_meter = utils.AverageMeter()
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss_meter.update(float(loss), batch.num_graphs)
        loss.backward()

        optimizer.step()
    return loss_meter.avg


def eval(args, model, device, loader, normalized_weight):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape).cpu())
        y_scores.append(pred.cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_scores = torch.cat(y_scores, dim = 0).numpy()

    roc_list = []
    weight = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
            weight.append(normalized_weight[i])

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    weight = np.array(weight)
    roc_list = np.array(roc_list)

    return weight.dot(roc_list)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'chembl_filtered', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    TokenMAEClf.add_args(parser)
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #Bunch of classification tasks
    if args.dataset == "chembl_filtered":
        num_tasks = 1310
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    loader = DataLoaderSL(0, dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, pin_memory=True)

    # gnn = GNN_v2(args.gnn_encoder_layer, args.gnn_emb_dim, True, JK=args.gnn_JK, drop_ratio=args.gnn_dropout, gnn_type=args.gnn_type, gnn_activation=args.gnn_activation)
    
    gnn = TokenMAEClf(args.freeze_token, args.gnn_token_layer, args.gnn_encoder_layer, args.gnn_emb_dim, args.gnn_JK, args.gnn_dropout, gnn_type=args.gnn_type,
    d_model=args.d_model, trans_encoder_layer=args.trans_encoder_layer, nhead=args.nhead, dim_feedforward=args.dim_feedforward, transformer_dropout=args.transformer_dropout, transformer_activation=args.transformer_activation, transformer_norm_input=args.transformer_norm_input, custom_trans=args.custom_trans, args=args)
    gnn = gnn.to(args.device)
    if not args.input_model_file == "":
        gnn.load_state_dict(torch.load(args.input_model_file), strict=False)

    #set up model
    model = GNN_graphpred(gnn, args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)  
    print(optimizer)

    torch.save(model.gnn.state_dict(), args.output_model_file)
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        print(train(args, model, device, loader, optimizer))
    torch.save(model.gnn.state_dict(), args.output_model_file)



if __name__ == "__main__":
    main()