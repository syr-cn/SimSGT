import argparse
from loader import MoleculeDataset
from dataloader import DataLoaderSL, DataLoaderSL_v2, DataLoaderSL_v3

import torch
import os
import torch.optim as optim
import utils
from tqdm import tqdm
from model import TokenMAE
from pos_enc.loader import MoleculeDataset_Eig_v2
from pos_enc.pos_enc import dataset_precomputing


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
    model = TokenMAE(args.gnn_encoder_layer, args.gnn_token_layer, args.gnn_decoder_layer, args.gnn_emb_dim, args.nonpara_tokenizer, args.gnn_JK, args.gnn_dropout, args.gnn_type,
    args.d_model, args.trans_encoder_layer, args.trans_decoder_layer, args.nhead, args.dim_feedforward, args.transformer_dropout, args.transformer_activation, args.transformer_norm_input, custom_trans=args.custom_trans, drop_mask_tokens=args.drop_mask_tokens, use_trans_decoder=args.use_trans_decoder, pe_type=args.pe_type, args=args)

    model = model.to(args.device)

    # set up optimizers
    params = [
        {'params': model.tokenizer.parameters(), 'lr': args.lr * args.tlr_scale},
        {'params': model.encoder.parameters(), 'lr': args.lr},
        {'params': model.decoder.parameters(), 'lr': args.lr},
        ]
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.decay)

    if args.model_file is not None and args.resume_epoch > 0:
        print(f'Loading model from {args.model_file}')
        state_dict = torch.load(args.model_file, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f'Loading optim from {args.optim_file}')
        state_dict = torch.load(args.optim_file, map_location=torch.device('cpu'))
        optimizer.load_state_dict(state_dict)


    for epoch in range(1+args.resume_epoch, args.resume_epoch+args.epochs+1):
        import time
        t1 = time.time()
        train_mae(args, model, loader, optimizer, epoch)
        print(f'{time.time()-t1:.2f}s')
        if epoch % args.save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_{epoch}.pth'))
        
    ## Save a final model
    torch.save(model.state_dict(), os.path.join(log_dir, f'model_{epoch}.pth'))
    torch.save(optimizer.state_dict(), os.path.join(log_dir, f'optim_{epoch}.pth'))

if __name__ == "__main__":
    main()
