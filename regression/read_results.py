import os
import re
import argparse
import numpy as np
from collections import defaultdict


def print_std(accs, stds, categories, append_mean=False):
    category_line = ' '.join(categories)
    if append_mean:
        category_line += ' Mean'
    
    line = ''
    if append_mean:
        acc = sum(accs) / len(accs)
        line += '{:0.2f} '.format(acc)
        line += '{:0.2f} '.format(acc-67.01)

    if stds is None:
        for acc in accs:
            line += '{:0.2f} '.format(acc)
    else:
        for acc, std in zip(accs, stds):
            line += '{:0.2f}±{:0.2f} '.format(acc, std)
    
    print(category_line)
    print(line)


def read_bio(args):
    args.begin = max(1, args.begin)
    with open(args.path, 'r') as f:
        lines = f.readlines()
    lines = lines[args.begin-1:args.end]
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line) > 0]

    regex = re.compile('Seed-(\d); Epoch-(\d+); 0 ([.\d]+) ([.\d]+) ([.\d]+)')
    test_hard_acc_list = []
    for line in lines:
        result = regex.match(line)
        if result:
            seed = int(result.group(1))
            test_easy_acc = float(result.group(4)) * 100
            test_hard_acc = float(result.group(5)) * 100
            test_hard_acc_list.append(test_hard_acc)
    test_hard_acc_list = np.asarray(test_hard_acc_list)
    print(test_hard_acc_list)
    print('{:0.1f}±{:0.1f}'.format(test_hard_acc_list.mean(), test_hard_acc_list.std()))


def read_chem(args):
    args.begin = max(1, args.begin)
    with open(args.path, 'r') as f:
        lines = f.readlines()
        lines = lines[args.begin-1:args.end]
        lines = [line.strip().split() for line in lines]
        lines = [line for line in lines if len(line) > 0]

    model2lines = {}
    for line in lines:
        if len(line) == 5:
            dataset, model_path, seed, val_roc, test_roc = line
        elif len(line) == 4:
            dataset, seed, val_roc, test_roc = line
            model_path = ''
        elif len(line) == 3:
            dataset, seed, test_roc = line
            model_path = ''
        else:
            raise NotImplementedError
        if model_path not in model2lines:
            model2lines[model_path] = []
        model2lines[model_path].append(line)
    for model, lines in model2lines.items():
        dataset2rocs = {}
        for line in lines:
            if len(line) == 5:
                dataset, model_path, seed, val_roc, test_roc = line
            elif len(line) == 4:
                dataset, seed, val_roc, test_roc = line
                model_path = ''
            elif len(line) == 3:
                dataset, seed, test_roc = line
                val_roc = 0
                model_path = ''
            else:
                raise NotImplementedError
            test_roc = float(test_roc)
            if dataset in dataset2rocs:
                dataset2rocs[dataset][seed] = test_roc
            else:
                dataset2rocs[dataset] = {seed: test_roc}

        dataset2results = {}
        for dataset, rocs in dataset2rocs.items():
            rocs = np.asarray(list(rocs.values()))
            roc_mean = rocs.mean()
            roc_std = rocs.std()
            dataset2results[dataset] = (roc_mean, roc_std)
        
        dataset_list = ['BBBP', 'Tox21', 'ToxCast',
                        'SIDER', 'ClinTox', 'MUV', 'HIV', 'BACE']
        dataset_list = [dataset.lower() for dataset in dataset_list]
        rocs = [dataset2results.get(dataset, [0,0])[0]*100 for dataset in dataset_list]
        stds = [dataset2results.get(dataset, [0,0])[1]*100 for dataset in dataset_list]
        print(f'Performance for {model}')
        print_std(rocs, stds, dataset_list, True)


def read_chem_dir(args):
    log_dir = args.path
    acc_mean_list = []
    acc_std_list = []
    dataset_list = ['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace']
    for dataset in dataset_list:
        log_name = f'{args.scheme_prefix}_{dataset}_log_100.txt'
        log_path = os.path.join(log_dir, log_name)
        if not os.path.exists(log_path):
            acc_mean_list.append(0)
            acc_std_list.append(0)
            continue
        with open(log_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
        test_acc_list = []
        for line in lines:
            if len(line) < 3:
                continue
            if line[0] == 'Epoch' and line[1] == f'{args.epoch}:' and len(line) >= 4:
                test_acc = float(line[-1])
                test_acc_list.append(test_acc)
        if len(test_acc_list) == 0:
            test_acc_list = [0]
        
        test_acc_list = np.asarray(test_acc_list)
        test_acc_mean = np.mean(test_acc_list)
        acc_std = np.std(test_acc_list)
        test_acc_mean = round(100 * test_acc_mean, 1)
        acc_std = round(100 * acc_std, 1)
        acc_mean_list.append(test_acc_mean)
        acc_std_list.append(acc_std)
    print_std(acc_mean_list, acc_std_list, dataset_list, True)


def read_chem_dir_ratio(args):
    log_dir = args.path
    acc_mean_list = []
    acc_std_list = []
    dataset_list = ['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace']
    for dataset in dataset_list:
        log_name = f'{args.scheme_prefix}_{dataset}_log.txt'
        log_path = os.path.join(log_dir, log_name)

        if not os.path.exists(log_path):
            acc_mean_list.append(0)
            acc_std_list.append(0)
            continue
        else:
            epoch = int(args.ratio * 100)

        with open(log_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
        test_acc_list = []
        for line in lines:
            if len(line) < 3:
                continue
            if line[0] == 'Epoch' and line[1] == f'{epoch}:' and len(line) >= 4:
                test_acc = float(line[-1])
                test_acc_list.append(test_acc)
        if len(test_acc_list) == 0:
            test_acc_list = [0]
        test_acc_list = np.asarray(test_acc_list)
        test_acc_mean = np.mean(test_acc_list)
        acc_std = np.std(test_acc_list)
        test_acc_mean = round(100 * test_acc_mean, 1)
        acc_std = round(100 * acc_std, 1)
        acc_mean_list.append(test_acc_mean)
        acc_std_list.append(acc_std)
    print_std(acc_mean_list, acc_std_list, dataset_list, True)

def read_dir_reg(args):
    log_dir = args.path
    dataset_list = ['esol', 'lipophilicity', 'malaria', 'cep']
    dataset_dict = {dataset:[] for dataset in dataset_list}
    log_name = f'{args.scheme_prefix}.txt'
    log_path = os.path.join(log_dir, log_name)

    with open(log_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split() for line in lines]
    for line in lines:
        if len(line) < 3:
            continue
        dataset = line[0]
        dataset_dict[dataset].append(float(line[-1]))

    line = []
    avg = 0
    for dataset in dataset_list:
        if len(dataset_dict[dataset])==0:
            dataset_dict[dataset].append(0)
        test_acc_list = np.asarray(dataset_dict[dataset])
        test_acc_mean = np.mean(test_acc_list)
        avg += test_acc_mean
        acc_std = np.std(test_acc_list)
        line.append(f'{test_acc_mean:.3f}({acc_std:.3f})')
    avg /= len(dataset_list)
    line.append(f'{avg:.4f}')
    line = ' '.join(line)

    print('\t  '.join([*dataset_list, 'avg']))
    print(line)

def read_dta_metrics(args):
    log_dir = args.path
    dataset_list = ['davis', 'kiba']
    metrics = ['rmse', 'mse'] #, 'pearson', 'spearman', 'ci']
    dataset_dict = {dataset:{metric:[] for metric in metrics} for dataset in dataset_list}
    log_name = f'{args.scheme_prefix}.txt'
    log_path = os.path.join(log_dir, log_name)

    with open(log_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split() for line in lines]
    for line in lines:
        if len(line) < 3:
            continue
        dataset = line[0]
        for i, metric in enumerate(metrics):
            dataset_dict[dataset][metric].append(float(line[3+i]))
    
    # print('\t', '\t\t'.join(metrics), sep='')
    line = []
    avg = 0
    for dataset in dataset_list:
        # print(f'{dataset}: ', end='\t')
        for metric in metrics:
            if len(dataset_dict[dataset][metric])==0:
                test_acc_list = dataset_dict[dataset][metric]=[0]
            test_acc_list = np.asarray(dataset_dict[dataset][metric])
            test_acc_mean = np.mean(test_acc_list)
            acc_std = np.std(test_acc_list)
            if metric == 'mse':
                avg += test_acc_mean
            line.append(f'{test_acc_mean:.4f}({acc_std:.4f})')
    avg /= len(dataset_list)
    line.append(f'{avg:.4f}')
    print(' '.join(line))

class DtaRecoder():
    def __init__(self, metrics):
        self.best = 0
        self.dict = {
            'valid':dict(),
            'test':dict()
        }
        self.metrics = metrics
        
    def set(self, task, value):
        for idx, metric in enumerate(self.metrics):
            self.dict[task][metric]=value[idx]

    def get(self, task, metric):
        return self.dict[task][metric]

def read_dta_ratio(args):
    max_epoch = 500
    epochs = [int(0.1*i*max_epoch) for i in range(1, 11)]
    print('reading test score with best val performance')
    log_dir = args.path
    dataset_list = ['davis', 'kiba']
    metrics = ['rmse', 'mse']
    scores = {dataset:[] for dataset in dataset_list}

    for ds in dataset_list:
        log_name = f'{args.scheme_prefix}_{ds}_log.txt'
        log_path = os.path.join(log_dir, log_name)
        if not os.path.exists(log_path):
            continue
        with open(log_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
        for line in lines:
            if len(line) < 3:
                continue
            if line[0] == 'Runseed':
                if len(scores[ds])==0 or scores[ds][-1][-1].best!=0:
                    scores[ds].append([
                        DtaRecoder(metrics) for _ in range(max_epoch+1)
                    ])
                    scores[ds][-1][0].set('valid', [100 for _ in metrics])
                    scores[ds][-1][0].set('test', [100 for _ in metrics])

            if line[0] == 'Epoch':
                epoch = int(line[1].split('(')[0])
                metric_score = [float(i) for i in line[2:]]
                if 'valid' in line[1]:
                    scores[ds][-1][epoch].set('valid', metric_score)
                    last_epoch = scores[ds][-1][epoch-1].best
                    if metric_score[-1] < scores[ds][-1][last_epoch].get('valid', 'mse'):
                        scores[ds][-1][epoch].best = epoch
                    else:
                        scores[ds][-1][epoch].best = last_epoch
                elif 'test' in line[1]:
                    scores[ds][-1][epoch].set('test', metric_score)
    
    for epoch in epochs:
        line = [f'epoch {epoch}:\t']
        avg = 0
        for ds in dataset_list:
            for metric in metrics:
                if len(scores[ds])==0:
                    test_acc_list = np.asarray([0])
                else:
                    test_acc_list = np.asarray([seed[seed[epoch].best].get('test', metric) for seed in scores[ds]])
                test_acc_list = test_acc_list[test_acc_list<99]
                test_acc_mean = np.mean(test_acc_list)
                acc_std = np.std(test_acc_list)
                if metric == 'mse':
                    avg += test_acc_mean
                line.append(f'{test_acc_mean:.4f}({acc_std:.4f})')
        avg /= len(dataset_list)
        line.append(f'{avg:.4f}')
        print(' '.join(line))

def read_log_reg(args):
    print('reading test score with best val performance')
    log_dir = args.path
    dataset_list = ['esol', 'lipophilicity', 'malaria', 'cep']
    dataset_dict = {dataset:[] for dataset in dataset_list}

    for dataset in dataset_list:
        log_name = f'{args.scheme_prefix}_{dataset}_log.txt'
        log_path = os.path.join(log_dir, log_name)
        if not os.path.exists(log_path):
            continue
        with open(log_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
        scores = []
        for line in lines:
            if len(line) < 3:
                continue
            if line[0] == 'Runseed':
                if len(scores)==0 or scores[-1]!= []:
                    scores.append([])
            if line[0] == 'Epoch':
                scores[-1].append((float(line[-2]), float(line[-1])))
        for score in scores:
            idx = 0
            for i, (val, test) in enumerate(score):
                if val < score[idx][0]:
                    idx = i
            dataset_dict[dataset].append(score[idx][1])
    
    line = []
    avg = 0
    for dataset in dataset_list:
        if len(dataset_dict[dataset])==0:
            dataset_dict[dataset].append(0)
        test_acc_list = np.asarray(dataset_dict[dataset])
        test_acc_mean = np.mean(test_acc_list)
        avg += test_acc_mean
        acc_std = np.std(test_acc_list)
        line.append(f'{test_acc_mean:.3f}({acc_std:.3f})')
    avg /= len(dataset_list)
    line.append(f'{avg:.4f}')
    line = ' '.join(line)

    print('\t'.join([*dataset_list, 'avg']))
    print(line)

def read_reg_ratio(args):
    # reading test score with best val performance
    max_epochs = 1000
    log_dir = args.path
    dataset_list = ['esol', 'lipophilicity', 'malaria', 'cep']
    dataset_dict = {dataset:defaultdict(list) for dataset in dataset_list}

    for dataset in dataset_list:
        log_name = f'{args.scheme_prefix}_{dataset}_log.txt'
        log_path = os.path.join(log_dir, log_name)
        if not os.path.exists(log_path):
            continue
        with open(log_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
        scores = []
        for line in lines:
            if len(line) < 3:
                continue
            if line[0] == 'Runseed':
                if len(scores)==0 or scores[-1]!= []:
                    scores.append([])
            if line[0] == 'Epoch':
                scores[-1].append((float(line[-2]), float(line[-1])))
        for score in scores:
            idx = 0
            for i, (val, test) in enumerate(score):
                if val < score[idx][0]:
                    idx = i
                if (i+1)%(max_epochs//10) == 0:
                    dataset_dict[dataset][i+1].append(score[idx][1])

    for epoch in range(max_epochs//10, max_epochs+1, max_epochs//10):
        line = []
        avg = 0
        for dataset in dataset_list:
            if len(dataset_dict[dataset][epoch])==0:
                dataset_dict[dataset][epoch].append(0)
            test_acc_list = np.asarray(dataset_dict[dataset][epoch])
            test_acc_mean = np.mean(test_acc_list)
            avg += test_acc_mean
            acc_std = np.std(test_acc_list)
            line.append(f'{test_acc_mean:.3f}({acc_std:.3f})')
        avg /= len(dataset_list)
        line.append(f'{avg:.4f}')
        line = ' '.join(line)

        print('\t'.join([f'epoch {epoch}', *dataset_list, 'avg']))
        print(line)


def read_chem_val(args):
    log_dir = args.path
    acc_mean_list = []
    acc_std_list = []
    dataset_list = ['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace']
    for dataset in dataset_list:
        log_name = f'{args.scheme_prefix}_{dataset}_log.txt'
        log_path = os.path.join(log_dir, log_name)
        if not os.path.exists(log_path):
            acc_mean_list.append(0)
            acc_std_list.append(0)
            continue
        with open(log_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
        val_acc_list = []
        test_acc_list = []
        best_test_acc_list = []
        for line in lines:
            if len(line) < 3:
                continue
            # print(line)
            if line[0] == 'Runseed':
                if len(val_acc_list) > 0:
                    val_acc_list = np.asarray(val_acc_list)
                    test_acc_list = np.asarray(test_acc_list)
                    best_test_acc_list.append(test_acc_list[val_acc_list.argmax()])
                val_acc_list = []
                test_acc_list = []
                
            if line[0] == 'Epoch' and len(line) >= 4:
                # epoch = float(line[1])
                val_acc = float(line[-2])
                test_acc = float(line[-1])
                val_acc_list.append(val_acc)
                test_acc_list.append(test_acc)
        
        if len(val_acc_list) > 0:
            val_acc_list = np.asarray(val_acc_list)
            test_acc_list = np.asarray(test_acc_list)
            best_test_acc_list.append(test_acc_list[val_acc_list.argmax()])
            print(best_test_acc_list)
        
        print(dataset, len(best_test_acc_list))
        best_test_acc_list = np.asarray(best_test_acc_list)
        test_acc_mean = best_test_acc_list.mean()
        acc_std = np.std(best_test_acc_list)
        test_acc_mean = round(100 * test_acc_mean, 1)
        acc_std = round(100 * acc_std, 1)
        acc_mean_list.append(test_acc_mean)
        acc_std_list.append(acc_std)
    print_std(acc_mean_list, acc_std_list, dataset_list, True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Reader')
    parser.add_argument('--dataset', type=str, default='chem',
                        help='The output comes from which dataset?')
    parser.add_argument('--path', type=str, help='Performance Log')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--scheme_prefix', type=str, default='linear')
    parser.add_argument('--begin', type=int, default=0, help='the line in which you begin to count')
    parser.add_argument('--end', type=int, default=100000, help='the line in which you stop to count')
    args = parser.parse_args()
    path_type = args.path.split('/')[1][:3]
    if args.mode == None:
        if path_type == 'reg':
            args.mode = 'read_log_reg'
        elif path_type == 'dta':
            args.mode = 'read_dta_ratio'
        elif path_type == 'mol':
            args.mode = 'log'
            args.path = os.path.join(args.path, 'result.log')
        elif path_type == 'run':
            args.mode = 'log'
            args.path = os.path.join(args.path, 'linear.txt')
        else:
            args.mode = 'read_dir_ratio'

    if args.dataset == 'bio':
        read_bio(args)
    elif args.dataset == 'chem':
        if args.mode == 'dir':
            read_chem_dir(args)
        elif args.mode == 'log':
            read_chem(args)
        elif args.mode == 'read_dir':
            print(f'Log: {args.path} -- {args.scheme_prefix}')
            for epoch in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                args.epoch = epoch
                print(f'Epoch {epoch}:', end='')
                read_chem_dir(args)
        elif args.mode == 'read_dir_ratio':
            print(f'Log: {args.path} -- {args.scheme_prefix}')
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                args.ratio = ratio
                print(f'Ratio {ratio}:', end='')
                read_chem_dir_ratio(args)
        elif args.mode == 'read_dir_reg':
            read_dir_reg(args)
        elif args.mode == 'read_log_reg':
            read_log_reg(args)
        elif args.mode == 'read_reg_ratio':
            read_reg_ratio(args)
        elif args.mode == 'read_dta':
            read_dta_metrics(args)
        elif args.mode == 'read_dta_ratio':
            read_dta_ratio(args)
        elif args.mode == 'val':
            read_chem_val(args)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError()
