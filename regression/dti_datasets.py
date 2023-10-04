import os

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat

from loader import mol_to_graph_data_obj_simple

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


class MoleculeProteinDataset(InMemoryDataset):
    def __init__(self, root, dataset):
        self.root = root
        self.dataset = dataset
        super(MoleculeProteinDataset, self).__init__(root)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return f'{self.dataset}.csv'

    @property
    def processed_file_names(self):
        return f'{self.dataset}.pt'

    def download(self):
        pass

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data
    
    def process(self):
        # load csv
        input_df = pd.read_csv(self.raw_paths[0])
        # Molecules pre-process
        smiles_list = input_df['compound_iso_smiles']
        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in rdkit_mol_objs_list]
        preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in preprocessed_rdkit_mol_objs_list]
        assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
        assert len(smiles_list) == len(preprocessed_smiles_list)
        smiles_list, rdkit_mol_objs = preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list

        # Proteins pre-process
        protein_list = input_df['target_sequence'].tolist()
        protein_list = [seq_cat(t) for t in protein_list]

        # label pre-process
        label_list = input_df['affinity'].tolist()

        data_list = []
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol != None:
                data = mol_to_graph_data_obj_simple(rdkit_mol, extra_feature=True)
                data.id = torch.tensor([i])
                data.y = torch.FloatTensor([label_list[i]])
                data.target = torch.LongTensor(protein_list[i])
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    # def __len__(self):
    #     return len(self.label_list)
