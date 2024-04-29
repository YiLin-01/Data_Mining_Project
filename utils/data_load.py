import os
import numpy as np
import pandas as pd
import torch
import scipy.io
from scipy.io import loadmat
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

from utils.maskable_list import MaskableList


def replace_nan_in_mat(data_dir):
    """
    Replace NaN in .mat files with 0

    Args: 
        data_dir: directory of .mat files
    """
    for filename in os.listdir(data_dir):
        if filename.endswith('.mat'):
            data = loadmat(f'{data_dir}/{filename}')
            for key in data:
                if not key.startswith('__'):
                    data[key] = np.nan_to_num(data[key])
                    scipy.io.savemat(f'{data_dir}/{filename}', data)

def load_adj_label_HCP(data_dir, label_dir, modality='dti'):
    """
    load adjacency matrix from raw matlab (G, N, N) & load the label from csv file (G,).
    We need to make them a correct pair!

    Args:
        data_dir: root dir to åsave the dataset
        dataset: name of the dataset
        modality: 'dti' by default (structural MRI, SMRI) or 'func' (FMRI)
        is_dup: Whether the label file has duplicated subject

    returns:
        adj: (G, N, N), G is the graph for cleaned subjects
        y: (G,)
    """

    # Load the dict of adj matrix (including all the subjects such dirty ones, duplicated ones)

    # dirty currently
    # Don't no if need to transpose the adj matrix
    # adj = adj_dict[adj_key].transpose((2, 1, 0))  # np.narray (# dirty graphs/substances, # node, # node)

    # Find the Label from the csv file
    #label_file_name = os.path.splitext(label_dir)[0]
    # demo_file = f'{label_file_name}' + ('_dup.csv' if Is_dup else '.csv')
    data_dir = f'{data_dir}/functional_network_132_0' if modality == 'func' else f'{data_dir}/structural_network_132_0'
    demo_df = pd.read_csv(label_dir)
    select_cols = ['Subject', 'Race', 'Gender', 'Age']
    demo_df = demo_df[select_cols]
    
    adj_list = []
    label_list = []
    # Get the corresponding adj
    if modality == 'func':
        for idx, sub in enumerate(demo_df['Subject']):
            if not os.path.isfile(f'{data_dir}/{sub}.mat'):
                print(f'{sub}.mat does not exist in the dataset path!')
                continue

            ##dirty currently
            ## adj = adj_dict[adj_key].transpose((2, 1, 0))  # np.narray (# dirty graphs/substances, # node, # node)
            ## adj = extract_knn(a1, args.top_k) # when args.top_k == 0, return the original a1

            ## Get the corresponding adj from adj path
            ## adj = adj[sub_list, :, :]
            adj_dict = loadmat(f'{data_dir}/{sub}')
            adj_key = list(adj_dict.keys())[-1]
            adj = torch.Tensor(adj_dict[adj_key])

            # Make the negative weights positive for fmri
            adj[adj < 0] = -adj[adj < 0]
            mask = torch.isnan(adj)
            adj[mask] = 0
            adj_list.append(adj)
            # Process labels
            gender = demo_df.loc[idx, 'Gender']
            label = 0 if gender == 'M' else 1
            label_list.append(label)

    elif modality == 'dti':
        for idx, sub in enumerate(demo_df['Subject']):
            if not os.path.isfile(f'{data_dir}/{sub}_density.txt'):
                print(f'{sub}_density.txt does not exist in the dataset path!')
                continue
            
            adj_nparray = np.loadtxt(f'{data_dir}/{sub}_density.txt')
            adj = torch.Tensor(adj_nparray)

            # Min-Max normalization for smri
            adj = (adj - adj.min()) / (adj.max() - adj.min())
            adj_list.append(adj)
            # Process labels
            gender = demo_df.loc[idx, 'Gender']
            label = 0 if gender == 'M' else 1
            label_list.append(label)
    else:
        raise Exception("The modality is not supported")
    
    adj_tensor = torch.stack(adj_list) if adj_list else torch.empty(0)
    y = torch.tensor(label_list, dtype=torch.long) if label_list else torch.empty(0, dtype=torch.long)
    # y = torch.tensor(label_df).long().flatten()  # labels for all the graph (#graph)

    return adj_tensor, y


def generate_node_features(adj, node_feature_type='Gassuin', f_dim=None):
    """
    Args:
        adj: weighted adjacency matrix 
        node_feature_type: The way to generate the node features
        f_dim: optional, not all of type will use it
    returns:
        x1: node feature (#graph, #node, #f_dim)
    """
    n_graph = adj.shape[0]
    n_node = adj.shape[1]
    if f_dim is None:
        f_dim = n_node
    if node_feature_type == 'Gassuin' or 'gassuin':  # standard normalization (mu=0,std=1)
        torch.manual_seed(42) 
        x1 = torch.normal(mean=0., std=1., size=(n_graph, n_node, f_dim))
        # x1 = torch.randn(n_graph, n_node, f_dim) # same
    elif node_feature_type == 'adj':  # use the adjacency matrix as the node feature
        x1 = adj.float()
    else:
        raise Exception("The type to generate node features is not supported")
    
    return x1


# mark the edge index in tiled n^2 vector
def generate_edge_flag(n_node, edge_index):
    """
    Args:
        edge_index: (2, E) tensor, E is the number of edges 

    returns:
        edge_flag: (n_node, n_node) 1-dim tensor. if non-zero, edge_flag is True, else False, bool Tensor
    """
    edge_flag = np.full((n_node ** 2,), False)
    n_edge = edge_index.shape[1]
    for i in range(n_edge):
        source = edge_index[0, i]
        target = edge_index[1, i]
        new_index = source * n_node + target
        edge_flag[new_index] = True

    edge_flag = torch.from_numpy(edge_flag)
    return edge_flag


def dense_adj_2_COO(adj):
    """
    Args:
        adj: dense adjacency matrix

    returns:
        ls_edge_index is list of the COO connectivity，
        ls_edge_weight is list the weight of each connected edge (scalar),
        ls_edge_flag is the list of flattened binary adj matrix.

        ls_edge_index: 0/1 entries, [(2, E), ...], where len is G
        ls_edge_weight: [(E), ...], where len is G
        ls_edge_flag: binary entries, [(N*N, ), ..], where len is G
    """

    ls_edge_index = []
    ls_edge_weight = []  # Get it from the adj matrix
    ls_edge_flag = []
    n_graph = adj.shape[0]
    n_node = adj.shape[1]

    for i in range(n_graph): 
        edge_index, edge_weight = dense_to_sparse(adj[i])  # (2, E), (E)
        edge_flag = generate_edge_flag(n_node, edge_index) # For each graph, flattened binary adj matrix (N*N)/flattened edge_index

        ls_edge_index.append(edge_index)
        ls_edge_weight.append(edge_weight)
        ls_edge_flag.append(edge_flag)

    return ls_edge_index, ls_edge_weight, ls_edge_flag


def build_dataset(x, ls_edge_index, ls_edge_weight, ls_edge_flag, y, adj):
    """
    Args: (all of the elements are tensor)
        x: node feature tensor (G, N, F)
        ls_edge_index: 0/1 entries, [(2, E), ...], where len is G
        ls_edge_weight: [(E), ...], where len is G
        ls_edge_flag: 0/1 entries, [(N*N, ), ..], where len is G, bool Tensor
        y: (G,), labels of all the graphs
        adj: (G, N, N), G is the graph for cleaned subjects
    returns:
        PyG.Data list: [Data(x=[], edge_index=[], edge_weight=[], edge_flag=[], y=[], adj=[]), ...]
    """
    data_list = MaskableList([])
    n_graph = y.shape[0]
    for idx in range(n_graph):
        data = Data(x=x[idx], edge_index=ls_edge_index[idx],
                    edge_weight=ls_edge_weight[idx], edge_attr=ls_edge_weight[idx],
                    edge_flag=ls_edge_flag[idx], y=y[idx], adj=adj[idx])
        data_list.append(data)
    return data_list



if __name__ == '__main__':
    import time
    start = time.time()
    # sequence for data loading
    modality = 'dti'

    label_dir = '/home/lihanzhao/SparceGNN4Brain/HCP_Data/filtered_annotations.csv'
    
    f_dim = None
    node_feature_type = 'gaussian'
    print("Getting adj & y...")
    adj, y = load_adj_label_HCP('HCP_Data', label_dir, modality)
    print("Done.")
    print('adj size:', adj.shape)
    print('label size:', y.shape)
    print()

    print("Getting node features...")
    x = generate_node_features(adj=adj, node_feature_type=node_feature_type, f_dim=f_dim)
    print("Done.")

    print("Get edge characters in COO format...")
    ls_edge_index, ls_edge_weight, ls_edge_flag = dense_adj_2_COO(adj)

    print("Getting data list for dataset...")
    data_list = build_dataset(x, ls_edge_index, ls_edge_weight, ls_edge_flag, y, adj)
    print("Done.")
    
    end = time.time()
    print("The total time for data loading is {} minutes.".format((end-start)/60))
    print("Dataset is ready.")