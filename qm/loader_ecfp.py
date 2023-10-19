from loader import *

ECFP_DIM = 1024
ECFP_RADIUS = 2

def ecfp2tensor(x):
    n = x.shape[0]
    ecfp4 = torch.zeros(n, ECFP_DIM)
    for atom in range(n):
        for l in range(ECFP_RADIUS+1):
            ecfp_dim = int(x[atom][l])
            ecfp4[atom][ecfp_dim] += 1
    return ecfp4

class MoleculeDataset_ecfp(InMemoryDataset):
    def __init__(self,
                 root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset_ecfp, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
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
        data_smiles_list = []
        data_list = []

        input_path = self.raw_paths[0]
        input_df = pd.read_csv(input_path, sep=',', compression='gzip',
                                dtype='str')
        smiles_list = list(input_df['smiles'])
        zinc_id_list = list(input_df['zinc_id'])
        for i in range(len(smiles_list)):
            s = smiles_list[i]
            # each example contains a single species
            rdkit_mol = AllChem.MolFromSmiles(s)
            if rdkit_mol != None:  # ignore invalid mol objects
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                info = dict()
                ecfp4 = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol,ECFP_RADIUS,ECFP_DIM,bitInfo=info)
                n = data.x.shape[0]
                ecfp4 = torch.zeros(n, ECFP_RADIUS+1)
                for k, v in info.items():
                    for pos in v:
                        ecfp4[pos[0]][pos[1]] = k
                data.ecfp4 = ecfp4
                # manually add mol id
                id = int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
                data.id = torch.tensor(
                    [id])  # id here is zinc id value, stripped of
                # leading zeros
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
