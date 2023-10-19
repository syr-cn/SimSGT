import argparse
from loader import MoleculeDataset
from dataloader import DataLoaderSL, DataLoaderSL_v2, DataLoaderSL_v3

import torch
import os
import torch.optim as optim
import utils
from tqdm import tqdm
from model import TokenMAE, NoEdgeTokenizer, ParaTokenizer, GNNDecoderComplete_v2, GNNDecoder_v2, NoEdgeDecoder
from pos_enc.loader import MoleculeDataset_Eig_v2
from pos_enc.pos_enc import dataset_precomputing
import time


num_atom_type = 120
ogb_num_atom_type = 119

class GraphMAEPretrainer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.no_edge_tokenizer:
            self.encoder = NoEdgeTokenizer(args)
        else:
            self.encoder = ParaTokenizer(args)

        self.num_classes = ogb_num_atom_type if args.complete_feature else num_atom_type
        GNNDecoder = GNNDecoderComplete_v2 if args.complete_feature else GNNDecoder_v2
        if args.tk_no_edge_decoder:
            GNNDecoder = NoEdgeDecoder
        self.decoder = GNNDecoder(
            args.gnn_emb_dim,
            args.gnn_emb_dim,
            self.num_classes,
            args.tk_decoder_layers,
            gnn_activation=args.tk_activation,
            gnn_type=args.tk_gnn_type,
            gnn_jk=args.tk_decoder_JK,
            use_input_norm=args.decoder_input_norm,
            trans_layer=0
        )
        self.loss = args.loss
        self.remask = args.tk_decoder_remask

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        g_nodes = torch.nn.functional.one_hot(x[data.mask_tokens][:, 0], self.num_classes)

        h = self.encoder(data.x_masked, edge_index, edge_attr)
        g_pred = self.decoder(h, edge_index, edge_attr, data.mask_tokens, data.batch, None, remask=self.remask)
        g_pred = g_pred[data.mask_tokens]

        if self.loss == 'mse':
            loss = self.mse_loss(g_nodes, g_pred)
        elif self.loss == 'sce':
            loss = self.sce_loss(g_nodes, g_pred)
        else:
            raise NotImplementedError()
        return loss
    
    def mse_loss(self, x, y):
        loss = ((x - y) ** 2).mean()
        return loss

    def sce_loss(self, x, y, alpha: float=1):
        x = F.normalize(x, p=2.0, dim=-1) # shape = [N, D]
        y = F.normalize(y, p=2.0, dim=-1) # shape = [N, D]
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss



def train(args, model, loader, optimizer, epoch):
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
    parser.add_argument('--subset', action='store_true', default=False)
    parser.add_argument('--block_mask', action='store_true', default=False)
    parser.add_argument('--block_size', type=int, default=2)
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
    
    dataset_name = args.dataset
    # set up dataset and transform function.
    if args.pe_type != 'none':
        dataset = MoleculeDataset_Eig_v2('dataset_eig/' + dataset_name, dataset=dataset_name, args=args)
        dataset = dataset_precomputing(args, dataset)
    else:
        dataset = MoleculeDataset("dataset/" + dataset_name, dataset=dataset_name)

    if args.subset:
        utils.set_seed(1001)
        train_size = int(0.9 * len(dataset))
        train_set = torch.randperm(len(dataset))[:train_size]
        dataset = dataset[train_set]

    if args.block_mask:
        loader = DataLoaderSL_v3(args.mask_rate, args.block_size, dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, pin_memory=True)
    else:
        loader = DataLoaderSL(args.mask_rate, dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, pin_memory=True)

    # model = TokenMAE(args.encoder_layer, args.token_layer, args.decoder_layer, args.emb_dim, args.nonpara_tokenizer, gnn_JK = args.JK, gnn_drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type,)
    model = GraphMAEPretrainer(args)
    model = model.to(args.device)

    # set up optimizers
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1+args.resume_epoch, args.resume_epoch+args.epochs+1):
        t1 = time.time()
        train(args, model, loader, optimizer, epoch)
        if epoch % args.save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_{epoch}.pth'))
        t2 = time.time()
        print(f'epoch {epoch}: {t2-t1}s.')
        
    ## Save a final model
    torch.save(model.encoder.state_dict(), args.tokenizer_path)
    torch.save(model.encoder.state_dict(), os.path.join(log_dir, f'model_{epoch}.pth'))
    torch.save(optimizer.state_dict(), os.path.join(log_dir, f'optim_{epoch}.pth'))

if __name__ == "__main__":
    main()
