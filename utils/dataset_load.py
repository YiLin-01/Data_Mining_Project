import numpy as np
import torch
from scipy.io import loadmat
import os
import pandas as pd
from numpy import linalg as LA
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from utils.utils import MaskableList
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data


########## feature generation ##################
# for LDP node features
def LDP(g, key='deg'):
    x = np.zeros([len(g.nodes()), 5])

    deg_dict = dict(nx.degree(g))
    for n in g.nodes():
        g.nodes[n][key] = deg_dict[n]

    for i in g.nodes():
        nodes = g[i].keys()

        nbrs_deg = [g.nodes[j][key] for j in nodes]

        if len(nbrs_deg) != 0:
            x[i] = [
                np.mean(nbrs_deg),
                np.min(nbrs_deg),
                np.max(nbrs_deg),
                np.std(nbrs_deg),
                np.sum(nbrs_deg)
            ]
    return x

# for degree_bin node features
def binning(a, n_bins=10):
    n_graphs = a.shape[0]
    n_nodes = a.shape[1]
    _, bins = np.histogram(a, n_bins)
    binned = np.digitize(a, bins)
    binned = binned.reshape(-1, 1)
    enc = OneHotEncoder()
    return enc.fit_transform(binned).toarray().reshape(n_graphs, n_nodes, -1).astype(np.float32)


########## data loading ##################
def load_adj_label_oasis(data_dir, dataset='oasis', modality='dti', Is_dup=False):
    """
    load adjacency matrix from raw matlab (G, N, N) & load the label from csv file (G,).
    We need to make them a correct pair!

    Args:
        data_dir: root dir to save the dataset
        dataset: name of the dataset
        modality: 'dti' by default (structural MRI, SMRI) or 'func' (FMRI)
        is_dup: Whether the label file has duplicated subject

    returns:
        adj: (G, N, N), G is the graph for cleaned subjects
        y: (G,)
    """

    # Load the dict of adj matrix (including all the subjects such dirty ones, duplicated ones)
    adj_dict = loadmat(f'{data_dir}/{dataset}_data.mat')  # 'datasets/oasis_data.mat'; .mat is not necessary

    if modality == 'dti':
        adj_key = 'normsc_oas'
    elif modality == 'func':
        adj_key = 'rssc_oas'
    else:
        raise Exception("The modality is not supported!")

    # dirty currently
    adj = adj_dict[adj_key].transpose((2, 1, 0))  # np.narray (# dirty graphs/substances, # node, # node)
    # adj = extract_knn(a1, args.top_k) # when args.top_k == 0, return the original a1

    # TODO(NOTE): Get the cleaned adj matrix
    demo_file = f'{dataset}_labels_bin'
    if Is_dup:
        demo_file += '_dup.csv'
    else:
        demo_file += '.csv'
    demo_path = os.path.join(data_dir, demo_file)
    demo_df = pd.read_csv(demo_path)
    select_cols = ['SUBJECT#', 'Subject', 'Gender', 'Primary Diagnosis']
    demo_df = demo_df[select_cols]
    sub_idx_narray = demo_df['SUBJECT#'].to_numpy() - 1  # idx of sub starts from 1, but idx of adj starts from 0
    sub_idx_list = sub_idx_narray.tolist()

    # Get the corresponding connectivity in adj
    adj = adj[sub_idx_list, :, :]
    adj = torch.Tensor(adj)

    # Make the negative weights positive for fmri
    if modality == 'func':
        adj[adj < 0] = -adj[adj < 0]

    # TODO(NOTE): Get the labels from demo_df
    label_df = demo_df['Primary Diagnosis'].to_frame()

    # Make the label numerical (binary)
    hc_mask = (label_df['Primary Diagnosis'] == 'Cognitively normal')
    ad_mask = (label_df['Primary Diagnosis'] == 'AD Dementia')

    label_df.loc[hc_mask, 'Primary Diagnosis'] = 0
    label_df.loc[ad_mask, 'Primary Diagnosis'] = 1

    labels = label_df.to_numpy(dtype='int64')
    y = torch.tensor(labels).long().flatten()  # labels for all the graph (#graph)

    # for sample in adj:
    #     nonzero = np.count_nonzero(sample)
    #     print(f'Non-zero: {nonzero} out of {sample.size}. percentage {nonzero/sample.size*100:.2f}')

    return adj, y


def load_adj_label_oasis_gender(data_dir, dataset='oasis', modality='dti', Is_dup=False):
    """
    load adjacency matrix from raw matlab (G, N, N) & load the label from csv file (G,2) including the gender info.
    We need to make them a correct pair!

    Args:
        data_dir: root dir to save the dataset
        dataset: name of the dataset
        modality: 'dti' by default (structural MRI, SMRI) or 'func' (FMRI)
        is_dup: Whether the label file has duplicated subject

    returns:
        adj: (G, N, N), G is the graph for cleaned subjects
        label_df: dataframe, ['Label', 'Gender']. (G,2)
    """

    # Load the dict of adj matrix (including all the subjects such dirty ones, duplicated ones)
    adj_dict = loadmat(f'{data_dir}/{dataset}_data.mat')  # 'datasets/oasis_data.mat'; .mat is not necessary

    if modality == 'dti':
        adj_key = 'normsc_oas'
    elif modality == 'func':
        adj_key = 'rssc_oas'
    else:
        raise Exception("The modality is not supported!")

    # dirty currently
    adj = adj_dict[adj_key].transpose((2, 1, 0))  # np.narray (# dirty graphs/substances, # node, # node)
    # adj = extract_knn(a1, args.top_k) # when args.top_k == 0, return the original a1

    # TODO(NOTE): Get the cleaned adj matrix
    demo_file = f'{dataset}_labels_bin'
    if Is_dup:
        demo_file += '_dup.csv'
    else:
        demo_file += '.csv'
    demo_path = os.path.join(data_dir, demo_file)
    demo_df = pd.read_csv(demo_path)
    select_cols = ['SUBJECT#', 'Subject', 'Gender', 'Primary Diagnosis']
    demo_df = demo_df[select_cols]

    sub_idx_narray = demo_df['SUBJECT#'].to_numpy() - 1  # idx of sub starts from 1, but idx of adj starts from 0
    sub_idx_list = sub_idx_narray.tolist()

    # Get the corresponding connectivity in adj
    adj = adj[sub_idx_list, :, :]
    adj = torch.Tensor(adj)

    # Make the negative weights positive for fmri
    if modality == 'func':
        adj[adj < 0] = -adj[adj < 0]

    # TODO(NOTE): Get the y from demo_df
    y_df = demo_df['Primary Diagnosis'].to_frame()

    # Make the label numerical (binary)
    hc_mask = (y_df['Primary Diagnosis'] == 'Cognitively normal')
    ad_mask = (y_df['Primary Diagnosis'] == 'AD Dementia')

    y_df.loc[hc_mask, 'Primary Diagnosis'] = 0
    y_df.loc[ad_mask, 'Primary Diagnosis'] = 1

    # TODO(NOTE): New added
    y_df = y_df.rename({'Primary Diagnosis': 'Label'}, axis='columns')

    gender_df = demo_df['Gender'].to_frame()
    gender_df = gender_df.rename({'Gender': 'Gender'}, axis='columns')

    frames = [y_df, gender_df]
    label_df = pd.concat(frames, axis=1, join='inner')
    label_df = label_df.reset_index(drop=True)

    # y = y_df.to_numpy(dtype='int64')
    # y = torch.tensor(y).long().flatten()  # labels for all the graph (#graph)

    # for sample in adj:
    #     nonzero = np.count_nonzero(sample)
    #     print(f'Non-zero: {nonzero} out of {sample.size}. percentage {nonzero/sample.size*100:.2f}')

    return adj, label_df


def load_adj_label_qced(data_dir, dataset='qced', modality='dti', Is_dup=False):
    """
    load adjacency matrix from raw matlab (G, N, N) & load the label from csv file (G,).
    We need to make them a correct pair!

    Args:
        data_dir: root dir to save the dataset
        dataset: name of the dataset
        modality: 'dti' by default (structural MRI, SMRI) or 'func' (FMRI)
        Is_dup: Whether the label file has duplicated subject

    returns:
        adj: (G, N, N), G is the graph for cleaned subjects
        y: (G,), labels of all the graphs
    """

    # Load the dict of adj matrix (including all the subjects such dirty ones, duplicated ones)
    adj_dict = loadmat(f'{data_dir}/{dataset}_data.mat')  # 'datasets/oasis_data.mat'; .mat is not necessary

    if modality == 'dti':
        adj_key = 'structural_network'
    elif modality == 'func':
        adj_key = 'functional_network'
    else:
        raise Exception("The modality is not supported!")

    # dirty currently
    adj = adj_dict[adj_key].transpose((2, 1, 0))  # np.narray (# dirty graphs/substances, # node, # node)
    # adj = extract_knn(a1, args.top_k) # when args.top_k == 0, return the original a1

    # TODO(NOTE): Get the cleaned adj matrix
    demo_file = f'{dataset}_labels_bin'
    if Is_dup:
        demo_file += '_dup.csv'
    else:
        demo_file += '.csv'
    demo_path = os.path.join(data_dir, demo_file)
    demo_df = pd.read_csv(demo_path)
    select_cols = ['index', 'ID', 'sub', 'Sex', 'Group']
    demo_df = demo_df[select_cols]
    sub_idx_narray = demo_df['index'].to_numpy() - 1  # idx of sub starts from 1, but idx of adj starts from 0
    sub_idx_list = sub_idx_narray.tolist()

    # Get the corresponding connectivity in adj
    adj = adj[sub_idx_list, :, :]
    adj = torch.Tensor(adj)

    # handle nan for fmri
    if modality == 'func':
        mask = torch.isnan(adj)
        adj[mask] = 0

    if modality == 'dti':  # Min-Max normalization for smri
        adj = (adj - adj.min()) / (adj.max() - adj.min())
    elif modality == 'func':  # Make the negative weights positive for fmri
        adj[adj < 0] = -adj[adj < 0]

    # TODO(NOTE): Get the labels from demo_df
    label_df = demo_df['Group'].to_frame()

    # Make the label numerical (binary)
    hc_mask = (label_df['Group'] == 'Cognitively normal')
    ad_mask = (label_df['Group'] == 'AD Dementia')

    label_df.loc[hc_mask, 'Group'] = 0
    label_df.loc[ad_mask, 'Group'] = 1

    labels = label_df.to_numpy(dtype='int64')
    y = torch.tensor(labels).long().flatten()  # labels for all the graph (#graph)

    # for sample in adj:
    #     nonzero = np.count_nonzero(sample)
    #     print(f'Non-zero: {nonzero} out of {sample.size}. percentage {nonzero/sample.size*100:.2f}')

    return adj, y


def load_adj_label_qced_gender(data_dir, dataset='qced', modality='dti', Is_dup=False): 
    """
    load adjacency matrix from raw matlab (G, N, N) & load the label from csv file (G,2) including the gender info.
    We need to make them a correct pair!

    Args:
        data_dir: root dir to save the dataset
        dataset: name of the dataset
        modality: 'dti' by default (structural MRI, SMRI) or 'func' (FMRI)
        Is_dup: Whether the label file has duplicated subject

    returns:
        adj: (G, N, N), G is the graph for cleaned subjects
        label_df: dataframe, ['Label', 'Gender']. (G,2)
    """

    # Load the dict of adj matrix (including all the subjects such dirty ones, duplicated ones)
    adj_dict = loadmat(f'{data_dir}/{dataset}_data.mat')  # 'datasets/oasis_data.mat'; .mat is not necessary

    if modality == 'dti':
        adj_key = 'structural_network'
    elif modality == 'func':
        adj_key = 'functional_network'
    else:
        raise Exception("The modality is not supported!")

    # dirty currently
    adj = adj_dict[adj_key].transpose((2, 1, 0))  # np.narray (# dirty graphs/substances, # node, # node)
    # adj = extract_knn(a1, args.top_k) # when args.top_k == 0, return the original a1

    # TODO(NOTE): Get the cleaned adj matrix
    demo_file = f'{dataset}_labels_bin'
    if Is_dup:
        demo_file += '_dup.csv'
    else:
        demo_file += '.csv'
    demo_path = os.path.join(data_dir, demo_file)
    demo_df = pd.read_csv(demo_path)
    select_cols = ['index', 'ID', 'sub', 'Sex', 'Group']
    demo_df = demo_df[select_cols]
    sub_idx_narray = demo_df['index'].to_numpy() - 1  # idx of sub starts from 1, but idx of adj starts from 0
    sub_idx_list = sub_idx_narray.tolist()

    # Get the corresponding connectivity in adj
    adj = adj[sub_idx_list, :, :]
    adj = torch.Tensor(adj)

    # handle nan for fmri
    if modality == 'func':
        mask = torch.isnan(adj)
        adj[mask] = 0

    if modality == 'dti':  # Min-Max normalization for smri
        adj = (adj - adj.min()) / (adj.max() - adj.min())
    elif modality == 'func':  # Make the negative weights positive for fmri
        adj[adj < 0] = -adj[adj < 0]

    # TODO(NOTE): Get the labels from demo_df
    y_df = demo_df['Group'].to_frame()

    # Make the label numerical (binary)
    hc_mask = (y_df['Group'] == 'Cognitively normal')
    ad_mask = (y_df['Group'] == 'AD Dementia')

    y_df.loc[hc_mask, 'Group'] = 0
    y_df.loc[ad_mask, 'Group'] = 1

    # TODO(NOTE): New added
    y_df = y_df.rename({'Group': 'Label'}, axis='columns')

    gender_df = demo_df['Sex'].to_frame()
    gender_df = gender_df.rename({'Sex': 'Gender'}, axis='columns')

    frames = [y_df, gender_df]
    label_df = pd.concat(frames, axis=1, join='inner')
    label_df = label_df.reset_index(drop=True)

    # y = y_df.to_numpy(dtype='int64')
    # y = torch.tensor(labels).long().flatten()  # labels for all the graph (#graph)

    # for sample in adj:
    #     nonzero = np.count_nonzero(sample)
    #     print(f'Non-zero: {nonzero} out of {sample.size}. percentage {nonzero/sample.size*100:.2f}')

    return adj, label_df


def generate_node_features(adj, node_feature_type='identity', f_dim=None):
    """
    Args:
        adj: weighted adjacency matrix (G, N, N), G is the graph for cleaned subjects
        node_feature_type: The way to generate the node features
        f_dim: optional, not all of type will use it
    returns:
        x1: node feature (#graph, #node, #f_dim)
    """
    n_graph = adj.shape[0]
    n_node = adj.shape[1]
    if f_dim is None:
        f_dim = n_node

    # construct node features X
    if node_feature_type == 'identity':
        x1 = torch.ones(n_graph, n_node, f_dim)
    elif node_feature_type == 'onehot':   #  one-hot
        # (#graph, #node, #f_dim), where the feature is one hot vector, #f_dim=#nodes
        # for each graph, it is a dialog matrix, so for each row (node), it is one-hot vector
        x = torch.cat([torch.diag(torch.ones(n_node))] * n_graph).reshape([n_graph, n_node, -1])
        x1 = x.clone()  # (#graph, #node, #f_dim=#nodes)

    elif node_feature_type == 'degree':
        # use degree to represent its feature
        a1b = (adj != 0).float()  # 0./1. tensor with (#graph, #node, #node)
        x1 = a1b.sum(dim=2, keepdim=True)  # (#graph, #node, #f_dim=1)

    elif node_feature_type == 'adj':  # use the adj matrix directly
        x1 = adj.float()  # (#graph, #node, #nodes)

    elif node_feature_type == 'LDP':  # degree profile
        a1b = (adj != 0).float()  # 0./1. tensor with (#graph, # node, # node)
        x1 = []
        for i in range(n_graph):
            x1.append(LDP(nx.from_numpy_array(a1b[i].numpy())))

    elif node_feature_type == 'eigen':
        _, x = LA.eig(adj.numpy())  # where is x1?
        x1 = x  # we add

    elif node_feature_type == 'degree_bin':
        a1b = (adj != 0).float()
        x1 = binning(a1b.sum(dim=2))  # (#graph, # node, # 1)

    elif node_feature_type == 'gaussian':  # standard normalization (mu=0,std=1)
        x1 = torch.normal(mean=0., std=1., size=(n_graph, n_node, f_dim))
        # x1 = torch.randn(n_graph, n_node, f_dim) # equivalent
    else:
        raise Exception("The type to generate node features is not supported")

    # elif node_feature_type == 'node2vec':
    #     X = np.load(f'./{args.dataset_name}_{args.modality}.emb', allow_pickle=True).astype(np.float32)
    #     x1 = torch.from_numpy(X)

    x1 = torch.Tensor(x1).float()
    return x1


# mark the edge index in tiled n^2 vector
def generate_edge_flag(n_node, edge_index):
    """
    Args:
        edge_index: (2, E)

    returns:
        edge_flag: (n_node*n_node, ), 1-dim, tensor. if non-zero, edge_flag is True, else False, bool Tensor
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
        adj: dense adjacency matrix, (G, N, N)

    returns:
        ls_edge_index is list of the COO connectivity, ls_edge_weight is list the weight of each connected edge (scalar),
        ls_edge_flag is the list of flattened binary adj matrix.
        All the elements in the list is tensor

        ls_edge_index: 0/1 entries, [(2, E), ...], where len is G
        ls_edge_weight: [(E), ...], where len is G
        ls_edge_flag: binary entries, [(N*N, ), ..], where len is G
    """
    ls_edge_index = []
    ls_edge_weight = []  # Get it from the adj matrix
    ls_edge_flag = []
    n_graph = adj.shape[0]
    n_node = adj.shape[1]
    for i in range(n_graph):  # For each graph
        edge_index, edge_weight = dense_to_sparse(adj[i])  # (2, E), (E)
        # print("I am here. {}".format(i))
        # input(binary, 0/1): (2, E) -> output(binary, 0/1): (N*N), still 1-dim
        edge_flag = generate_edge_flag(n_node, edge_index)  # For each graph, flattened binary adj matrix (N*N)/flattened edge_index

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

    """
    data_list = MaskableList([])
    n_graph = y.shape[0]
    for idx in range(n_graph):
        data = Data(x=x[idx], edge_index=ls_edge_index[idx],
                    edge_weight=ls_edge_weight[idx], edge_attr=ls_edge_weight[idx],
                    edge_flag=ls_edge_flag[idx], y=y[idx], adj=adj[idx])
        data_list.append(data)

    return data_list


def build_dataset_gender(x, ls_edge_index, ls_edge_weight, ls_edge_flag, label_df, adj):
    """
    Args: (all of the elements are tensor)
        x: node feature tensor (G, N, F)
        ls_edge_index: 0/1 entries, [(2, E), ...], where len is G
        ls_edge_weight: [(E), ...], where len is G
        ls_edge_flag: 0/1 entries, [(N*N, ), ..], where len is G, bool Tensor
        label_df: (G,2), including labels and gender info of all the graphs. ['Label', 'Gender']
        adj: (G, N, N), G is the graph for cleaned subjects
    returns:

    """

    data_list = MaskableList([])
    # print(label_df)
    y_df = label_df['Label']
    y = y_df.to_numpy(dtype='int64')
    y = torch.tensor(y).long().flatten()
    n_graph = y.shape[0]

    gender_df = label_df['Gender']

    for idx in range(n_graph):
        data = Data(x=x[idx], edge_index=ls_edge_index[idx],
                    edge_weight=ls_edge_weight[idx], edge_attr=ls_edge_weight[idx],
                    edge_flag=ls_edge_flag[idx], y=y[idx], adj=adj[idx], group=gender_df.iloc[idx])
        data_list.append(data)

    return data_list


if __name__ == '__main__':
    import time
    start = time.time()

    # sequence for data loading
    data_dir = 'datasets'
    dataset = 'qced'  # 'oasis'  #
    modality = 'dti'  # 'func'  #
    Is_dup = False
    f_dim = None  # use the N for it
    node_feature_type = 'gaussian'
    Is_dup = True  # False

    if dataset == 'oasis':
        adj, y = load_adj_label_oasis(data_dir=data_dir, dataset=dataset, modality=modality, Is_dup=Is_dup)
    elif dataset == 'qced':
        adj, y = load_adj_label_qced(data_dir=data_dir, dataset=dataset, modality=modality, Is_dup=Is_dup)
    else:
        raise Exception("The dataset is not supported")
    print("Get adj & y")

    x = generate_node_features(adj=adj, node_feature_type=node_feature_type, f_dim=f_dim)
    print("get node features")

    ls_edge_index, ls_edge_weight, ls_edge_flag = dense_adj_2_COO(adj)
    print("get edge characters in COO format")

    data_list = build_dataset(x, ls_edge_index, ls_edge_weight, ls_edge_flag, y, adj)
    print("get data list for dataset")

    end = time.time()

    print("The total time for data loading is {} minutes".format((end-start)/60))

    print(adj.shape)
    print(y.shape)
    print(data_list[0])