from loader import *
import rdkit
import rdkit.Chem as Chem
from tqdm import tqdm
from collections import defaultdict
from random import randint

PATT = {
    'HETEROATOM': '[!#6]',
    'DOUBLE_TRIPLE_BOND': '*=,#*',
    'ACETAL': '[CX4]([O,N,S])[O,N,S]'
}
PATT = {k: Chem.MolFromSmarts(v) for k, v in PATT.items()}

FG_NUM = 513
FG_DIM = 20

def fg2tensor(x, y):
    assert x.shape == y.shape
    n = x.shape[0]
    dim = x.shape[1]
    motif2atom = defaultdict(list)
    motif2fg = {}
    for atom in range(n):
        for l in range(dim):
            fg_id = int(x[atom][l])
            if fg_id<0:
                break
            m_id = int(y[atom][l])
            if m_id not in motif2fg:
                motif2fg[m_id] = fg_id
            motif2atom[mid].append(atom)
    fg = []
    mat = torch.zeros(len(motif2fg), n)
    for k, v in motif2atom.items():
        for atom in v:
            mat[len(fg)][atom]=1
        fg.append(motif2fg[k])
    fg = torch.tensor(fg)
    return fg, mat

def get_fg_set(mol):
    """
    Identify FGs and convert to SMILES
    Args:
        mol:
    Returns: a set of FG's SMILES
    """
    fgs = []  # Function Groups

    # <editor-fold desc="identify and merge rings">
    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]  # get simple rings
    flag = True  # flag == False: no rings can be merged
    while flag:
        flag = False
        for i in range(len(rings)):
            if len(rings[i]) == 0: continue
            for j in range(i + 1, len(rings)):
                shared_atoms = rings[i] & rings[j]
                if len(shared_atoms) > 2:
                    rings[i].update(rings[j])
                    rings[j] = set()
                    flag = True
    rings = [r for r in rings if len(r) > 0]
    # </editor-fold>

    # <editor-fold desc="identify functional atoms and merge connected ones">
    marks = set()
    for patt in PATT.values():  # mark functional atoms
        for sub in mol.GetSubstructMatches(patt):
            marks.update(sub)
    atom2fg = [[] for _ in range(mol.GetNumAtoms())]  # atom2fg[i]: list of i-th atom's FG idx
    for atom in marks:  # init: each marked atom is a FG
        fgs.append({atom})
        atom2fg[atom] = [len(fgs)-1]
    for bond in mol.GetBonds():  # merge FGs
        if bond.IsInRing(): continue
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in marks and a2 in marks:  # a marked atom should only belong to a FG, if atoms are both marked, merge their FGs into a FG
            assert a1 != a2
            assert len(atom2fg[a1]) == 1 and len(atom2fg[a2]) == 1
            # merge a2' FG to a1's FG
            fgs[atom2fg[a1][0]].update(fgs[atom2fg[a2][0]])
            fgs[atom2fg[a2][0]] = set()
            atom2fg[a2] = atom2fg[a1]
        elif a1 in marks:  # only one atom is marked, add neighbour atom to its FG as its environment
            assert len(atom2fg[a1]) == 1
            # add a2 to a1's FG
            fgs[atom2fg[a1][0]].add(a2)
            atom2fg[a2].extend(atom2fg[a1])
        elif a2 in marks:
            # add a1 to a2's FG
            assert len(atom2fg[a2]) == 1
            fgs[atom2fg[a2][0]].add(a1)
            atom2fg[a1].extend(atom2fg[a2])
        else:  # both atoms are unmarked, i.e. a trivial C-C single bond
            # add single bond to fgs
            fgs.append({a1, a2})
            atom2fg[a1].append(len(fgs)-1)
            atom2fg[a2].append(len(fgs)-1)

    tmp = []
    for fg in fgs:
        if len(fg) == 0: continue
        if len(fg) == 1 and mol.GetAtomWithIdx(list(fg)[0]).IsInRing(): continue
        tmp.append(fg)
    fgs = tmp
    # </editor-fold>

    fgs.extend(rings)  # final FGs: rings + FGs (not in rings)

    fg_smiles = set()
    fg_record = defaultdict(list)
    for fg in fgs:
        # fg_smiles.add(Chem.MolFragmentToSmiles(mol, fg))
        fg_record[Chem.MolFragmentToSmiles(mol, fg)].extend(list(fg))
    

    return fg_record


class MoleculeDataset_RelMole(InMemoryDataset):
    def __init__(self,
                root,
                #data = None,
                #slices = None,
                transform=None,
                pre_transform=None,
                pre_filter=None,
                dataset='zinc_standard_agent_relmole',
                empty=False):
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset_RelMole, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                'No download allowed')

    def process(self):
        data_list = []
        if self.dataset == 'zinc_standard_agent_relmole':
            input_path = self.raw_paths[0]
            corpus_path = os.path.join(self.root, 'fg_corpus.txt')
            input_df = pd.read_csv(input_path, sep=',', compression='gzip',
                                   dtype='str')
            smiles_list = list(input_df['smiles'])
            with open(corpus_path, 'r') as f:
                fg_corpus = f.read().splitlines()
            
            for s in smiles_list:
                rdkit_mol = AllChem.MolFromSmiles(s)
                if rdkit_mol != None:  # ignore invalid mol objects
                    data = mol_to_graph_data_obj_simple(rdkit_mol)

                    fg_record = get_fg_set(rdkit_mol)
                    vocab_id = [[] for _ in range(data.x.shape[0])]
                    motif_id = [[] for _ in range(data.x.shape[0])]
                    for fg, atoms in fg_record.items():
                        mid = randint(0, int(1e9+7))
                        try:
                            idx = fg_corpus.index(fg)
                        except:
                            idx = FG_NUM-1
                        for atom in atoms:
                            vocab_id[atom].append(idx)
                            motif_id[atom].append(mid)
                    for atom in vocab_id:
                        while len(atom)<FG_DIM:
                            atom.append(-1)
                    for atom in motif_id:
                        while len(atom)<FG_DIM:
                            atom.append(-1)

                    data.vocab_id = torch.tensor(vocab_id)
                    data.motif_id = torch.tensor(motif_id)
                    data_list.append(data)
        else:
            raise ValueError('Invalid dataset name')

        print(f'length of datalist after processing: {len(data_list)}')
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
