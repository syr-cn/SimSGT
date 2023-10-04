import random
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.loader.dataloader import Collater

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 



# TODO(Bowen): more unittests
class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        
        self.num_chirality_tag = 3
        self.num_bond_direction = 3 


    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # ----------- graphMAE -----------
        atom_type = F.one_hot(data.mask_node_label[:, 0], num_classes=self.num_atom_type).float()
        atom_chirality = F.one_hot(data.mask_node_label[:, 1], num_classes=self.num_chirality_tag).float()
        # data.node_attr_label = torch.cat((atom_type,atom_chirality), dim=1)
        data.node_attr_label = atom_type

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

            edge_type = F.one_hot(data.mask_edge_label[:, 0], num_classes=self.num_edge_type).float()
            bond_direction = F.one_hot(data.mask_edge_label[:, 1], num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)
            # data.edge_attr_label = edge_type

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)



def drop_nodes(data, aug_ratio):
    '''
    This version returns nondrop nodes ids along with the function
    '''
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)
    idx_drop_set = set(idx_perm[:drop_num].tolist())
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    
    idx_dict = np.zeros((idx_nondrop[-1]+1,), dtype=np.int64)
    idx_dict[idx_nondrop] = np.arange(len(idx_nondrop), dtype=np.int64)

    edge_index = data.edge_index.numpy()

    edge_mask = []
    for n in range(edge_num):
        if not (edge_index[0, n] in idx_drop_set or edge_index[1, n] in idx_drop_set):
            edge_mask.append(n)
    edge_mask = np.asarray(edge_mask, dtype=np.int64)
    edge_index = idx_dict[edge_index[:, edge_mask]]
    try:
        data.edge_index = torch.from_numpy(edge_index) #.transpose(0, 1)
        data.x = data.x[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
        data.idx_nondrop = torch.from_numpy(idx_nondrop)
    except:
        data = data

    return data


class DropAtom:
    '''
    Drop Atom and keep a undropped version for decoder usage
    '''
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        
        self.num_chirality_tag = 3
        self.num_bond_direction = 3 


    def __call__(self, data):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """
        # sample x distinct atoms to be masked, based on mask rate. But
        # will sample at least 1 atom
        num_atoms = data.x.size()[0]
        sample_size = int(num_atoms * self.mask_rate + 1)
        masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # ----------- graphMAE -----------
        atom_type = F.one_hot(data.mask_node_label[:, 0], num_classes=self.num_atom_type).float()
        atom_chirality = F.one_hot(data.mask_node_label[:, 1], num_classes=self.num_chirality_tag).float()
        data.node_attr_label = atom_type

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])
        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)



class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                elif key  == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class DataLoaderMaskingPred(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, mask_rate=0.0, mask_edge=0.0, **kwargs):
        self._transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = mask_rate, mask_edge=mask_edge)
        super(DataLoaderMaskingPred, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs)
    
    def collate_fn(self, batches):
        batchs = [self._transform(x) for x in batches]
        batch = BatchMasking.from_data_list(batchs)
        
        ## add self loop
        #add self loops in the edge space
        edge_index, _ = add_self_loops(batch.edge_index, num_nodes = batch.x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(batch.x.size(0), 2, dtype=torch.long)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        edge_attr = torch.cat((batch.edge_attr, self_loop_attr), dim = 0)
        batch.edge_index = edge_index
        batch.edge_attr = edge_attr
        
        return batch



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
        batch.edge_index = edge_index
        batch.edge_attr = edge_attr
        return batch


def subgraph(data, sub_ratio):
    node_num = data.x.shape[0]
    subgraph_size = min(int(node_num * sub_ratio) + 1, node_num)
    neighbors = {i: [] for i in range(node_num)}
    edge_index_list = data.edge_index.T.tolist()
    for i, j in edge_index_list:
        neighbors[i].append(j)
    root = random.sample(range(node_num), 1)[0]
    sub_index = {root, }
    idx_neigh = set(neighbors[root]).difference([root])

    while len(sub_index) < subgraph_size:
        if len(sub_index) >= node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = random.sample(idx_neigh, k=1)[0]
        sub_index.add(sample_node)
        idx_neigh = idx_neigh.union(neighbors[sample_node])
        idx_neigh.difference_update(sub_index)

    # sub_index = list(sub_index)
    mask_index = [i for i in range(node_num) if i not in sub_index]
    data.mask_index = torch.LongTensor(mask_index)
    return data


class DataLoaderSL_v2(torch.utils.data.DataLoader):
    def __init__(
        self,
        mask_ratio, 
        subgraph_mask,
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
        self.subgraph_mask = subgraph_mask

    def collate_fn(self, graphs):
        if self.subgraph_mask:
            graphs = [subgraph(graph, 1-self.mask_ratio) for graph in graphs]
        batch = self._collater(graphs)
        if self.mask_ratio > 0:
            ## generate mask idx
            ptr = batch.ptr.tolist()
            if self.subgraph_mask:
                mask_idx = batch.mask_index.tolist()
                delattr(batch, 'mask_index')
            else:
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
            batch.x_dropped = batch.x[~mask_tokens]
            
            ## drop edges
            mask_idx = set(mask_idx)
            edge_unmask_idx = []
            edge_index_list = batch.edge_index.T.tolist()
            for idx, (i, j) in enumerate(edge_index_list):
                if i not in mask_idx and j not in mask_idx:
                    edge_unmask_idx.append(idx)
            edge_unmask_idx = torch.LongTensor(edge_unmask_idx)
            batch.edge_index_dropped = batch.edge_index[:, edge_unmask_idx]
            batch.edge_attr_dropped = batch.edge_attr[edge_unmask_idx]
            
            ## re-map edge index
            mapping = torch.full((batch.x.shape[0],), fill_value=-1, dtype=torch.long) # shape = [N]
            mapping[~mask_tokens] = torch.arange(batch.x_dropped.shape[0], dtype=torch.long)
            batch.edge_index_dropped = mapping[batch.edge_index_dropped]


        ## add self loop
        #add self loops in the edge space
        edge_index, _ = add_self_loops(batch.edge_index, num_nodes = batch.x.size(0))
        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(batch.x.size(0), 2, dtype=torch.long)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        edge_attr = torch.cat((batch.edge_attr, self_loop_attr), dim = 0)
        batch.edge_index = edge_index
        batch.edge_attr = edge_attr

        if self.mask_ratio > 0:
            ## add self loop for the dropped version
            edge_index_dropped, _ = add_self_loops(batch.edge_index_dropped, num_nodes = batch.x_dropped.size(0))
            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(batch.x_dropped.size(0), 2, dtype=torch.long)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            edge_attr_dropped = torch.cat((batch.edge_attr_dropped, self_loop_attr), dim = 0)
            batch.edge_index_dropped = edge_index_dropped
            batch.edge_attr_dropped = edge_attr_dropped
        return batch


class DataLoaderSL_v3(torch.utils.data.DataLoader):
    def __init__(
        self,
        mask_ratio, 
        block_size,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch = None,
        exclude_keys = None,
        **kwargs,
    ):
        self.block_size = block_size
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
        graphs = [block_mask(graph, self.mask_ratio, self.block_size) for graph in graphs]
        batch = self._collater(graphs)
        if self.mask_ratio > 0:
            ## generate mask idx
            # ptr = batch.ptr.tolist()
            mask_idx = batch.mask_index.tolist()
            delattr(batch, 'mask_index')
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
        batch.edge_index = edge_index
        batch.edge_attr = edge_attr
        return batch



def block_mask(data, mask_ratio, block_size=2):
    node_num = data.x.shape[0]
    mask_size = min(int(node_num * mask_ratio), node_num)
    neighbors = {i: [] for i in range(node_num)}
    edge_index_list = data.edge_index.T.tolist()
    for i, j in edge_index_list:
        neighbors[i].append(j)
    
    mask_index = set()
    cands = set(range(node_num))
    while len(mask_index) < mask_size:
        i = random.sample(cands, 1)[0]
        mask_index.add(i)
        cands.remove(i)
        neighs = cands.intersection(neighbors[i])
        neighs = random.sample(neighs, k=min(block_size, len(neighs)))
        for j in neighs:
            if len(mask_index) >= mask_size:
                break
            mask_index.add(j)
            cands.remove(j)
    
    mask_index = list(mask_index)
    mask_index.sort()
    data.mask_index = torch.LongTensor(mask_index)
    return data