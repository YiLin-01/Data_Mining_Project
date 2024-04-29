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

def load_adj_label_HCP(data_dir, label_dir, att = 'Gender', modality='dti'):
    """
    load adjacency matrix from raw matlab (G, N, N) & load the label from csv file (G,).
    We need to make them a correct pair!

    Args:
        data_dir: root dir to åsave the dataset
        label_dir: the path to the csv file containing the labels
        att: the attribute to be used as the label, 'Gender' by default
        modality: 'dti' by default (structural MRI, SMRI) or 'func' (FMRI)

    returns:
        adj: (G, N, N), G is the graph for cleaned subjects
        y: (G,)
    """

    if modality == 'func':
        data_dir = f'{data_dir}/functional_network_132_0'  
    elif modality == 'dti':
        data_dir = f'{data_dir}/structural_network_132_0'
    else:
        raise Exception("The modality is not supported")
    
    # gt
    select_cols = ['Subject', 'Race', 'Gender', 'Age']
    
    label_df = pd.read_csv(label_dir)
    label_df = label_df[select_cols]
    
    adj_list = []
    label_list = []

    if modality == 'func':
        adj_key = 'functional_network'
        for idx, sub in enumerate(label_df['Subject']):
            if not os.path.isfile(f'{data_dir}/{sub}.mat'):
                print(f'{sub}.mat does not exist in the dataset path!')
                continue

            adj_dict = loadmat(f'{data_dir}/{sub}')
            #     adj_key = list(adj_dict.keys())[-1]
            adj = torch.Tensor(adj_dict[adj_key])

            # Make the negative weights positive for fmri
            adj[adj < 0] = -adj[adj < 0]

            mask = torch.isnan(adj)
            adj[mask] = 0
            adj_list.append(adj)

            if att.lower() == 'race':
                race = label_df.loc[idx, 'Race']
                if race == 'White':
                    label = 0
                    label_list.append(label)
                elif race == 'Black or African Am.':
                    label = 1
                    label_list.append(label)
                else: # Asian, others
                    label = 2
                    label_list.append(label)
            elif att.lower() == 'gender':
                gender = label_df.loc[idx, 'Gender']
                label = 0 if gender == 'M' else 1
                label_list.append(label)
            else:
                raise Exception("The attribute is not supported")

    elif modality == 'dti':
        for idx, sub in enumerate(label_df['Subject']):
            if not os.path.isfile(f'{data_dir}/{sub}_density.txt'):
                print(f'{sub}_density.txt does not exist in the dataset path!')
                continue
            
            adj_nparray = np.loadtxt(f'{data_dir}/{sub}_density.txt')
            adj = torch.Tensor(adj_nparray)

            # Min-Max normalization for smri
            adj = (adj - adj.min()) / (adj.max() - adj.min())
            adj_list.append(adj)
            # Process labels
            gender = label_df.loc[idx, 'Gender']
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
    if node_feature_type.lower() == 'gassuin': 
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
        edge_flag: (n_node*n_node, ) 1-dim bool tensor. 
        if non-zero, edge_flag is True, else False
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

def graph_num(HCP_data_dir):
    '''
    Get the number of graphs in HCP dataset
    '''
    return len(os.listdir(HCP_data_dir))

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
    modality = 'func'
    att = 'Race'

    data_dir = '/home/lihanzhao/Documents/W&M/CSCI680-BrainProject/IBGNN/HCP_Data/'
    label_dir = '/home/lihanzhao/Documents/W&M/CSCI680-BrainProject/IBGNN/HCP_Data/filtered_annotations.csv'
    
    f_dim = None
    node_feature_type = 'gaussian'
    print("Getting adj & y...")
    adj, y = load_adj_label_HCP(data_dir, label_dir, att, modality)
    print("Done.\n")
    print('adj size:', adj.shape)
    print('label size:', y.shape)

    print("Getting node features...")
    x = generate_node_features(adj=adj, node_feature_type=node_feature_type, f_dim=f_dim)
    print("Done.")

    print("Get edge characters in COO format...")
    ls_edge_index, ls_edge_weight, ls_edge_flag = dense_adj_2_COO(adj)
    print('ls_edge_index size:',len(ls_edge_index))
    print('ls_edge_weight size:',len(ls_edge_weight))
    print('ls_edge_flag size:',len(ls_edge_flag))

    print("Getting data list for dataset...")
    data_list = build_dataset(x, ls_edge_index, ls_edge_weight, ls_edge_flag, y, adj)
    print("Done.")
    
    end = time.time()
    print("The total time for data loading is {} minutes.".format((end-start)/60))
    print("Dataset is ready.")