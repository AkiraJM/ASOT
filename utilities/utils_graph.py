import numpy as np
from torch_geometric.data import Dataset

def info_collect_graph_dataset(dataset : Dataset, if_extract_graphs = True):
    if if_extract_graphs:
        G = [_ for _ in dataset]
    else:
        G = None
    y = [_.y[0].item() for _ in dataset]

    num_graphs = len(dataset)
    # idx_to_y = []
    # for eachy in y:
    #     if eachy not in idx_to_y:
    #         idx_to_y.append(eachy)
    num_classes = dataset.num_classes
    # y_to_idx = {idx_to_y[_]:_ for _ in range(num_classes)}
    num_features = dataset[0].num_features
    num_nodes_list = np.array([_.num_nodes for _ in dataset])
    num_edges_list = np.array([_.num_edges for _ in dataset])

    info = {
        "G": G,
        "y": y,
        "num_graphs" : num_graphs,
        "num_features" : num_features,
        "num_classes" : num_classes,
        "max_num_nodes" : int(num_nodes_list.max()),
        "mean_num_nodes" : float(num_nodes_list.mean()),
        "min_num_nodes" : int(num_nodes_list.min()),
        "max_num_edges" : int(num_edges_list.max()),
        "mean_num_edges" : float(num_edges_list.mean()),
        "min_num_edges" : int(num_edges_list.min()),
        # "y2idx_map" : y_to_idx,
        # "idx2y_map" : idx_to_y,
    }

    return info