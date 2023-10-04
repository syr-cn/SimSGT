import os
import argparse
import subprocess
import utils
from multiprocessing.pool import ThreadPool
from tuning import TokenMAEClf
import copy

def generate_command(args):
    command = ''
    for k, v in args.__dict__.items():
        if isinstance(v, bool):
            if v == True:
                command += f'--{k} '
            else:
                continue
        elif v == None or v == '':
            continue
        else:
            command += f'--{k} {v} '
    return command


def run_tuning(args, tuning_dataset):
    # command = f'''
    # PYTHONPATH="./" python ./chem/tuning.py --model_file {args.model_file} --split "scaffold" --num_workers 2 --epochs {args.epochs} --name {args.name} --evaluation_mode linear --batch_size 32 --use_schedule --device {args.device} --num_seeds {args.num_seeds} --scheme_prefix {args.scheme_prefix} --mix_gnn --tuning_dataset {args.tuning_dataset}
    # '''
    command = f'python ./tuning.py {generate_command(args)} --tuning_dataset {tuning_dataset}'
    log_path = f'./results/{args.name}/tuning_print_log.txt'
    utils.print_time_info(f'Start Tuning {tuning_dataset}')
    with open(log_path, 'a') as f:
        output = subprocess.call(command, shell=True, stdout=f, stderr=f)
        utils.print_time_info(f'{tuning_dataset} subprocess output: {output}')
    utils.print_time_info(f'Finished tuning {tuning_dataset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of pre-training of graph neural networks')
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
    parser.add_argument('--model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for splitting dataset.")
    parser.add_argument('--split', type=str, default="scaffold",
                        help='Bio dataset: Random or species split; Chem dataset: random or scaffold or random_scaffold.')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers for dataset loading')
    parser.add_argument('--use_schedule', action="store_true",
                        default=False, help='Use learning rate scheduler?')
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--scheme_prefix', type=str, default='linear', help='The name for tuning logs.')
    parser.add_argument('--tuning_dataset', type=str, default=None,
                        help='Used only for CHEM dataset. The dataset used for fine-tuning.')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation to speed up training.')
    parser.add_argument('--eval_train', action='store_true',
                        help='Evaluate the training dataset or not.')
    # number of random seeds
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--thread_num', type=int, default=2)
    parser.add_argument('--fast_eval', action='store_true', default=False)
    parser.add_argument('--run_remain', action='store_true', default=False)
    parser.add_argument('--force_noschedule', action='store_true', default=False)
    parser.add_argument('--freeze_epochs', type=int, default=0)
    parser.add_argument('--use_gnn2', action='store_true', default=False)
    TokenMAEClf.add_args(parser)
    args = parser.parse_args()
    
    if args.run_remain:
        assert not args.fast_eval
    if args.fast_eval:
        assert not args.run_remain

    if args.fast_eval:
        tunining_dataset_list = ["bbbp", "clintox", "hiv", "toxcast", "tox21", "sider", "bace", "muv"]
    else:
        # tunining_dataset_list = ["tox21", "sider", "bace"]
        tunining_dataset_list = ["muv", "bbbp", "clintox", "hiv", "toxcast", "tox21", "sider", "bace"]
    
    if not os.path.exists('results/{}'.format(args.name)):
        os.makedirs('results/{}'.format(args.name))
    print(str(args))
    # num = 2  # set to the number of workers you want (it defaults to the cpu count of your machine)
    tp = ThreadPool(args.thread_num)
    for tuning_dataset in tunining_dataset_list:
        args_clone = copy.deepcopy(args)
        if args.fast_eval:
            if tuning_dataset == 'muv':
                args_clone.num_seeds = 1
            else:
                args_clone.num_seeds = 7
        
        if args.run_remain:
            if tuning_dataset == 'muv':
                args_clone.num_seeds = 9
                args_clone.start_seed = 1
            else:
                args_clone.num_seeds = 3
                args_clone.start_seed = 7
        tp.apply_async(run_tuning, (args_clone, tuning_dataset))
    tp.close()
    tp.join()
    
    