import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix, to_dense_adj)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add


def dataset_precomputing(args, dataset):
    if not hasattr(dataset, 'data_list'):
        print('dataset has no data_list to be precomputed')
        return dataset

    # Precompute necessary statistics for positional encodings.
    start = time.perf_counter()
    print(f"Precomputing Positional Encoding statistics: "
                    f"SignNet for all graphs...")
    # Estimate directedness based on 10 graphs to save time.
    # is_undirected = all(d.is_undirected() for d in dataset[:10])
    is_undirected = False
    print(f"  ...estimated to be undirected: {is_undirected}")


    data_list = [compute_posenc_stats(
                        dataset.data_list[i],
                        pe_types=None,
                        is_undirected=is_undirected,
                        args=args
                )
                for i in tqdm(range(len(dataset.data_list)),
                            disable=False,
                            mininterval=10,
                            miniters=len(dataset.data_list)//20)]
    del dataset.data_list        

    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)
    elapsed = time.perf_counter() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                + f'{elapsed:.2f}'[-3:]
    print(f"Done! Took {timestr}")
    return dataset

def compute_posenc_stats(data, pe_types, is_undirected, args):
    # Eigen-decomposition with numpy for SignNet.
    norm_type = args.laplacian_norm.lower()
    if norm_type == 'none':
        norm_type = None

    data.EigVals, data.EigVecs = get_lap_decomp_stats(
        evals=data.EigVals, evects=data.EigVecs,
        max_freqs=args.max_freqs,
        eigvec_norm=args.eigvec_norm)
    # EigVals = (n, k, 1)
    # EigVecs = (n, k)
    return data

def get_lap_decomp_stats_from_precompute(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = evects.shape[0]  # Number of nodes, including disconnected nodes.

    # Normalize and pad eigen vectors.
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects[:, :max_freqs]

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan'))
    else:
        EigVals = evals[:, :max_freqs, :]

    return EigVals, EigVecs

def rw_precompute(data, args):
    kernel_times = args.kernel_times
    if args.kernel_times_func != 'none':
        kernel_times = list(range(*map(int, args.kernel_times_func.split('~'))))

    if hasattr(data, 'num_nodes'):
        N = data.num_nodes
    else:
        N = data.x.shape[0]

    rw_landing = get_rw_landing_probs(ksteps=kernel_times,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
    return rw_landing

def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing

def eigvec_precompute(data, args):
    eigvec_norm = args.eigvec_norm
    norm_type = args.raw_norm_type
    # Eigen Vectors Precomputing before pretraining
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    
    L = to_scipy_sparse_matrix(
            *get_laplacian(data.edge_index, normalization=norm_type,
                           num_nodes=N)
        )
    
    EigVals, EigVecs = eig_calc_cpu(L)

    EigVals = torch.from_numpy(EigVals)
    EigVecs = torch.from_numpy(EigVecs)
    return EigVals, EigVecs

def eig_calc_gpu(L):
    device = torch.device("cuda:0") 
    L=torch.from_numpy(L.toarray()).to(device)

    evals_sn, evects_sn = torch.linalg.eig(L)

    evals_sn = evals_sn.detach().cpu().numpy()
    evects_sn = evects_sn.detach().cpu().numpy()
    return evals_sn, evects_sn
    
def eig_calc_cpu(L):
    evals_sn, evects_sn = np.linalg.eigh(L.toarray())
    return evals_sn, evects_sn

def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """

    evals = evals.numpy()
    evects = evects.numpy()

    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    assert eigvec_norm == 'L2'
    evects = F.normalize(evects, p=2, dim=0)
    
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1)
    return EigVals, EigVecs


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs