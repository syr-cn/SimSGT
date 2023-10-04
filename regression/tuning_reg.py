import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from scheduler import PolynomialDecayLR
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import InMemoryDataset
import utils
import loader
from loader import MoleculeDataset
from pos_enc.loader import MoleculeDataset_Eig_v2
from dataloader import DataLoaderSL
from splitters import scaffold_split
from model import GNN_v2, GNNComplete_v2, get_activation, TokenMAE
from pos_enc.encoder import PosEncoder
import copy
import torch.nn.functional as F

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
        group.add_argument('--disable_trans', action='store_true', default=False)
        group.add_argument('--trans_pooling', type=str, default='none')
        group.add_argument('--complete_feature', action='store_true', default=False)

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
        self.pos_encoder = PosEncoder(args)
        self.freeze_token = freeze_token
        
        if args.complete_feature:
            self.tokenizer = GNNComplete_v2(1, emb_dim, True, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type, gnn_activation=args.gnn_activation)
            self.encoder = GNNComplete_v2(encoder_layer-1, emb_dim, False, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type, gnn_activation=args.gnn_activation, 
            d_model=d_model, trans_layer=trans_encoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, pe_dim=self.pos_encoder.pe_dim, trans_pooling=args.trans_pooling)
        else:
            self.tokenizer = GNN_v2(1, emb_dim, True, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type, gnn_activation=args.gnn_activation)
            self.encoder = GNN_v2(encoder_layer-1, emb_dim, False, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type, gnn_activation=args.gnn_activation, 
            d_model=d_model, trans_layer=trans_encoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, pe_dim=self.pos_encoder.pe_dim, trans_pooling=args.trans_pooling)

        self.gnn_act = get_activation(args.gnn_activation)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.freeze_token:
            with torch.no_grad():
                h = self.tokenizer(x, edge_index, edge_attr).detach()
        else:
            h = self.tokenizer(x, edge_index, edge_attr)
        
        pe_tokens = self.pos_encoder(data)
        h = self.encoder(self.gnn_act(h), edge_index, edge_attr, data.batch, pe_tokens=pe_tokens)

        return h


def load_dataset_chem(args):
    if args.pe_type != 'none':
        dataset = MoleculeDataset_Eig_v2(
            "./dataset_eig/" + args.tuning_dataset, dataset=args.tuning_dataset, args=args)
    else:
        dataset = MoleculeDataset(
            "./dataset/" + args.tuning_dataset, dataset=args.tuning_dataset)
    print(dataset)
    if args.split == "scaffold":
        smiles_list = pd.read_csv(
            './dataset/' + args.tuning_dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    else:
        raise ValueError("Invalid split option.")
    return train_dataset, valid_dataset, test_dataset

@torch.no_grad()
def choose_best_model_file(args, valid_dataset):
    args.dataset=args.tuning_dataset
    model = TokenMAE(args.gnn_encoder_layer, args.gnn_token_layer, args.gnn_decoder_layer, args.gnn_emb_dim, args.nonpara_tokenizer, args.gnn_JK, args.gnn_dropout, args.gnn_type,
    args.d_model, args.trans_encoder_layer, args.trans_decoder_layer, args.nhead, args.dim_feedforward, args.transformer_dropout, args.transformer_activation, args.transformer_norm_input, custom_trans=args.custom_trans, drop_mask_tokens=args.drop_mask_tokens, use_trans_decoder=args.use_trans_decoder, pe_type=args.pe_type, args=args)

    model = model.to(args.device)
    model.eval()

    best_score = None
    best_model_file = ''
    loss_meter = utils.AverageMeter()
    print(f'choosing model file from {args.model_file}', flush=True)
    for model_file in os.listdir(args.model_file):
        loss_meter.reset()
        try:
            model_state_dict = torch.load(os.path.join(args.model_file, model_file), map_location=lambda storage, loc: storage)
            model.load_state_dict(model_state_dict)
        except:
            print(f'bad model: {model_file}')
            continue
        if '_mr' in model_file:
            mr =int(model_file.split('_mr')[1][:2])
            mr /= 100
        else:
            mr = 0.35
        valid_loader = DataLoaderSL(mr, valid_dataset, batch_size=512, shuffle=False, num_workers=0)

        for batch in valid_loader:
            batch = batch.to(args.device)
            loss = float(model(batch))
            loss_meter.update(loss, batch.num_graphs)
        print(model_file, loss_meter.avg)
        if best_score == None or loss_meter.avg < best_score:
            best_score = loss_meter.avg
            best_model_file = model_file
    best_model_file = os.path.join(args.model_file, best_model_file)
    return best_model_file

def load_chem_gnn_model(args):
    gnn = TokenMAEClf(args.freeze_token, args.gnn_token_layer, args.gnn_encoder_layer, args.gnn_emb_dim, args.gnn_JK, 0, gnn_type=args.gnn_type,
    d_model=args.d_model, trans_encoder_layer=args.trans_encoder_layer, nhead=args.nhead, dim_feedforward=args.dim_feedforward, transformer_dropout=args.transformer_dropout, transformer_activation=args.transformer_activation, transformer_norm_input=args.transformer_norm_input, custom_trans=args.custom_trans, args=args)

    if os.path.exists(args.model_file):
        print("Loading model from %s." % args.model_file)
        model_state_dict = torch.load(
            args.model_file, map_location=lambda storage, loc: storage)
        gnn.load_state_dict(model_state_dict, strict=False)
    else:
        print('gnn model not found!')
    return gnn


@utils.timeit
def train_chem(args, model, device, loader, optimizer, scheduler, log_file, label_mean, label_std, freeze_gnn=False):
    criterion = nn.MSELoss()
    model.train()
    loss_meter = utils.AverageMeter()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch, freeze_gnn=freeze_gnn).to(torch.float64)
        y = batch.y.view(pred.shape).to(torch.float64)
        y = (y - label_mean) / (label_std + 1e-5)

        loss = criterion(pred, y)  # shape = [N, C]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(float(loss), pred.shape[0])

    if scheduler:
        scheduler.step()
    utils.write_log('Avg loss {:.4f}'.format(loss_meter.avg), log_file)
    return loss_meter.avg


@utils.timeit
@torch.no_grad()
def eval_chem(args, model, loader, label_mean, label_std, metric='rmse'):
    model.eval()
    y_true = []
    y_scores = []

    for batch in loader:
        batch = copy.copy(batch)
        batch = batch.to(args.device)
        pred = model(batch)
        pred = pred * label_std + label_mean
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if metric == 'rmse':
        rmse_score = utils.rmse(y_true, y_scores)
    elif metric == 'mse':
        rmse_score = utils.mse(y_true, y_scores)
    else:
        raise NotImplementedError

    return rmse_score


class GraphClf(nn.Module):
    def __init__(self, gnn, emb_dim, num_tasks, trans_pooling='none', args=None):
        super(GraphClf, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.trans_pooling = trans_pooling
        if args.clf_norm == 'layer':
            self.norm = nn.LayerNorm(emb_dim)
        elif args.clf_norm == 'l2':
            self.norm = lambda x:nn.functional.normalize(x, p = 2, dim = -1)
        else:
            self.norm = None
        self.linear = nn.Linear(emb_dim, num_tasks)
        # self.mlp = nn.Sequential(nn.Linear(emb_dim, 2*emb_dim), nn.ReLU(), nn.Linear(2*emb_dim, num_tasks))

    def forward(self, data, freeze_gnn=False):
        if freeze_gnn:
            with torch.no_grad():
                node_representation = self.gnn(data)
        else:
            node_representation = self.gnn(data)
        if self.trans_pooling == 'cls':
            graph_rep = node_representation
        else:
            graph_rep = self.pool(node_representation, data.batch)
        if self.norm:
            graph_rep = self.norm(graph_rep)
        return self.linear(graph_rep)


def get_eval_loader(dataset, args):
    dataloader = DataLoaderSL(0, dataset, batch_size=512, shuffle=False, num_workers=0)
    if len(dataset) > 512:
        return [batch for batch in dataloader]
    else:
        return [batch.to(args.device) for batch in dataloader]


def tuning_chem(args, train_dataset, valid_dataset, test_dataset, running_seeds):
    if args.force_noschedule:
        args.use_schedule = False
    train_eval_loader = get_eval_loader(train_dataset, args)
    val_loader = get_eval_loader(valid_dataset, args)
    test_loader = get_eval_loader(test_dataset, args)
    
    if args.norm_label:
        raw_name = os.listdir(f'dataset/{args.tuning_dataset}/raw/')[0]
        load_labels = getattr(loader, f'_load_{args.tuning_dataset}_dataset')
        _, _, labels = np.array(load_labels(f'dataset/{args.tuning_dataset}/raw/{raw_name}'))
        label_mean, label_std = np.mean(labels), np.std(labels)
        print(f'label_mean of {args.tuning_dataset}: {label_mean}')
        print(f'label_std of {args.tuning_dataset}: {label_std}')
    else:
        label_mean, label_std = 0, 1
        
    
    if os.path.isdir(args.model_file):
        args.model_file = choose_best_model_file(args, valid_dataset)
    for runseed in tqdm(running_seeds):
        args.runseed = runseed
        print('Training For Running Seed {}'.format(args.runseed))
        utils.set_seed(args.runseed, args.device)
        pin_memory = args.num_workers > 0
        train_loader = DataLoaderSL(0, train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)

        # set up model
        gnn = load_chem_gnn_model(args)
        if args.trans_encoder_layer > 0:
            out_dim = args.d_model
        elif args.gnn_JK == 'first_cat':
            out_dim = 2 * args.gnn_emb_dim
        elif args.gnn_JK == 'concat':
            out_dim = args.gnn_emb_dim * (args.num_layer + 1)
        else:
            out_dim = args.gnn_emb_dim
        model = GraphClf(gnn, out_dim, args.num_tasks, args.trans_pooling, args=args)
        model.to(args.device)
        # set up optimizer
        lr = args.lr

        params = [
            {'params': model.gnn.parameters(), 'lr': args.lr},
            {'params': model.linear.parameters(), 'lr': args.lr, 'weight_decay': args.linear_decay}
        ]
        if model.norm:
            params.append({'params': model.norm.parameters(), 'lr': args.lr})
        optimizer = optim.Adam(params, lr=lr, weight_decay=args.decay)
        
        if args.scheduler == 'steplr':
            scheduler = StepLR(optimizer, step_size=int(args.epochs/3), gamma=1/3)
        elif args.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=lr/10)
        else:
            scheduler = None
        print(optimizer)

        # naming tuning scheme
        log_file = 'results/{}/{}_{}_log.txt'.format(args.name, args.scheme_prefix, args.tuning_dataset)
        utils.write_log(str(args), log_file=log_file, print_=True)
        utils.write_log('Runseed {}; Epochs {}; WeightDecay {}; ModelFile {}.'.format(
            args.runseed, args.epochs, args.decay, args.model_file), log_file)

        train_roc_list = []
        val_roc_list = []
        test_roc_list = []

        best_idx = 0
        for epoch in range(1, args.freeze_epochs+args.epochs+1):
            print("====epoch " + str(epoch))
            freeze_gnn = epoch <= args.freeze_epochs
            train_chem(args, model, args.device, train_loader,
                       optimizer, scheduler, log_file, label_mean, label_std, freeze_gnn=freeze_gnn)

            if (not args.skip_evaluation) or (epoch == args.epochs):
                val_roc = eval_chem(args, model, val_loader, label_mean, label_std, metric=args.eval_metric)
                val_roc_list.append(val_roc)
                test_roc = eval_chem(args, model, test_loader, label_mean, label_std, metric=args.eval_metric)
                test_roc_list.append(test_roc)

                if val_roc < val_roc_list[best_idx]:
                    best_idx = epoch-1
                    # report epoch with best validation score
                if args.eval_train:
                    train_roc = eval_chem(args, model, train_eval_loader)
                    train_roc_list.append(train_roc)
                    utils.write_log('Epoch {}: {} {} {}'.format(
                        epoch, train_roc_list[-1], val_roc_list[-1], test_roc_list[-1]), log_file)
                else:
                    utils.write_log('Epoch {}: {} {}'.format(
                        epoch, val_roc_list[-1], test_roc_list[-1]), log_file)

        # train_roc = eval_chem(args, model, train_eval_loader)
        # utils.write_log('Final Epoch Train ROC: {}'.format(train_roc), log_file)
        result_path = 'results/{}/{}.txt'.format(args.name, args.scheme_prefix)
        with open(result_path, 'a') as f:
            line = '{} {} {} {} {}\n'.format(
                args.tuning_dataset, args.model_file, args.runseed, val_roc_list[best_idx], test_roc_list[best_idx])
            f.write(line)



def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for splitting dataset.")
    parser.add_argument('--split', type=str, default="scaffold",
                        help='Bio dataset: Random or species split; Chem dataset: random or scaffold or random_scaffold.')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers for dataset loading')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--scheme_prefix', type=str, default='linear', help='The name for tuning logs.')
    parser.add_argument('--tuning_dataset', type=str, default=None,
                        help='Used only for CHEM dataset. The dataset used for fine-tuning.')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation to speed up training.')
    parser.add_argument('--eval_train', action='store_true',
                        help='Evaluate the training dataset or not.')
    # number of random seeds
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--double_precision', action='store_true', default=False)
    parser.add_argument('--thread_num', type=int, default=2)
    parser.add_argument('--fast_eval', action='store_true', default=False)
    parser.add_argument('--run_remain', action='store_true', default=False)
    parser.add_argument('--force_noschedule', action='store_true', default=False)
    parser.add_argument("--clf_norm", type=str, default=None)
    parser.add_argument('--norm_label', action='store_true', default=False)
    parser.add_argument('--eval_metric', type=str, default='rmse', help='rmse loss for molecule regression tasks')
    TokenMAE.add_args(parser)
    parser.add_argument('--trans_pooling', type=str, default='none')
    parser.add_argument('--freeze_token', action='store_true', default=False)
    parser.add_argument('--freeze_epochs', type=int, default=0)
    parser.add_argument('--linear_decay', type=float, default=0.0)
    args = parser.parse_args()
    
    ## optimization for a specific server
    try:
        uname = os.getlogin()
        if uname[0] == 'z' and uname[-1] == 'n':
            args.num_workers = 0
    except:
        pass

    if args.double_precision:
        # use double precision to ensure reproducibility
        torch.set_default_tensor_type(torch.DoubleTensor)

    running_seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))

    if not os.path.exists('results/{}'.format(args.name)):
        os.makedirs('results/{}'.format(args.name))

    args.device = torch.device(
        "cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    datasets = ['esol', 'lipophilicity', 'malaria', 'cep']
    num_task_dict = {'esol':1, 'lipophilicity':1, 'malaria':1, 'cep':1}
    model_path = args.model_file
    if args.tuning_dataset is None:
        for dataset in datasets:
            args.tuning_dataset = dataset
            args.num_tasks = num_task_dict[dataset]
            train_dataset, valid_dataset, test_dataset = load_dataset_chem(args)
            args.model_file = model_path
            tuning_chem(args, train_dataset, valid_dataset,
                        test_dataset, running_seeds)
    else:
        args.num_tasks = num_task_dict[args.tuning_dataset]
        train_dataset, valid_dataset, test_dataset = load_dataset_chem(args)
        args.model_file = model_path
        tuning_chem(args, train_dataset, valid_dataset,
                    test_dataset, running_seeds)
    


if __name__ == "__main__":
    start = time.time()
    utils.print_time_info('Start Tuning')
    main()
    end = time.time()
    utils.print_time_info('End Tuning. Spend %d seconds' % (end - start))
