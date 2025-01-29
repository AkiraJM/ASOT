import os
import argparse

# numpy
import numpy as np

# pytorch
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_scatter import scatter_add

# sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

# other
import matplotlib.pyplot as plt
import time
from utilities.CacheLoader import Params_to_Filename, Check_and_Load, Save_Mat, Save_Mats, draw_mat, draw_mat2, err_compute, draw_curves_err, draw_basis
from utilities.utils_graph import info_collect_graph_dataset
from utilities.utils_all import init_other_result_dict, init_other_result_folders, init_SOT_result_dict, init_SOT_result_folders, json_write, json_load, record_reference
import copy
import json
from typing import List, Dict

# algorithm
from utilities.SubspaceOT import minmax_normalize, GNN_Null, GINConv_no_nn_multi, GNN_WWL
from utilities.SOT import SubspaceOT, SubspaceOT_kmeans, SubspaceOT_ML, pairwise_subspaceOT_distance, pairwise_wasserstein_distance_sinkhorn, asym_pairwise_subspaceOT_distance, pairwise_Minkowski_distance
from utilities.graphPCA import GINConv_no_nn, GCNConv_no_nn
from utilities.OTmethods import pairwise_wasserstein_distance, asym_pairwise_wasserstein_distance, pairwise_SOT_distance

# arguments
parser = argparse.ArgumentParser(description='Experimental code for anchor space optimal transport on tudataset.')

parser.add_argument('--json', type=str, default="params.json",
                        help='path of json file for parameter setting. (default: params.json)')
parser.add_argument('--exp_type', type=str, default="graph",
                        help='type of experiment. (default: graph)')
parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any. (default: 0)')
parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset. (default: MUTAG)')
parser.add_argument('--use_node_attr', type=bool, default=False,
                        help='if use node attributes. (default: False)')
parser.add_argument('--n_folds', type=int, default=10,
                        help='number of folds for dataset splitting. (default: 10)')
parser.add_argument('--random_state', type=int, default=0,
                        help='random_state. (default: 0)')
parser.add_argument('--repetitions', type=int, default=1,
                        help='number of repetitions. (default: 1)')
parser.add_argument('--if_single_fold', type=bool, default=True,
                        help='if only run a single fold. (default: True)')
parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for neural network training. (default: 0.01)')
parser.add_argument('--batch_size', type=int, default=0,
                        help='training batch size. If 0, use the dataset size. (default: 0)')
parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs. (default: 500)')
parser.add_argument('--num_basis', type=int, default=0,
                        help='number of basis vectors of SOT. if 0, use the maximum number of nodes. (default: 0)')
parser.add_argument('--ml_hiddens', type=int, default=0,
                        help='number of dimensions of the hidden layer of metric learning variant. if 0, use the number of feature dimensions. (default: 0)')
parser.add_argument('--OT_reg', type=float, default=0.1,
                        help='weight of regularization term of eOT. (default: 0.1)')
parser.add_argument('--OT_iter', type=int, default=50,
                        help='number of iterations of the Sinkhorn\'s algorithm. (default: 0.1)')
parser.add_argument('--norm', type=int, default=2,
                        help='order of norm for computing the Minkowski distance. (default: 2)')
parser.add_argument('--rc', type=float, default=1.0,
                        help='weight of the reconstruction loss. (default: 1.0)')
parser.add_argument('--lp', type=float, default=1.0,
                        help='weight of the l_p ball loss. (default: 1.0)')
parser.add_argument('--sp', type=float, default=1.0,
                        help='weight of the simplex constraint violation loss. (default: 1.0)')
parser.add_argument('--ml', type=float, default=10.0,
                        help='weight of the metric learning loss. (default: 10.0)')
parser.add_argument('--lasso_iters', type=int, default=20,
                        help='number of the sparse coding layers. (default: 20)')
parser.add_argument('--prep_scheme', type=str, default="GINConv", choices=["GINConv", "WWL_conti"],
                        help='scheme of the node feature preprocessing. (default: GINConv)')
parser.add_argument('--prep_layers', type=int, default=4,
                        help='number of layers of the preprocessing scheme. (default: 4)')
parser.add_argument('--baseline', type=str, default="EMD", choices=["EMD", "Sinkhorn"],
                        help='baseline method name. (default: EMD)')
parser.add_argument('--methods', nargs="+", type=str, default=["SOT_EMD", "SOT_Sinkhorn", "SOT_ML_EMD", "SOT_ML_Sinkhorn", "SOT_k_EMD", "SOT_k_Sinkhorn", "Sinkhorn", "sliced_Wass"],
                        help='names of methods. (default: SOT_EMD, SOT_Sinkhorn, SOT_ML_EMD, SOT_ML_Sinkhorn, SOT_k_EMD, SOT_k_Sinkhorn, Sinkhorn, sliced_Wass)')
parser.add_argument('--colors', nargs="+", type=str, default=["red", "orange", "blue", "deepskyblue", "purple", "magenta", "green", "lime"],
                        help='colors of curves. (default: red, orange, blue, deepskyblue, purple, magenta, green, lime)')
parser.add_argument('--n_projections', type=int, default=0,
                        help='number of projections of the sliced Wasserstein distance. If 0, use the number of feature dimensions. (default: 0)')
parser.add_argument('--r', type=int, default=0,
                        help='number of ranks of the low rank Sinkhorn. If 0, use the number of feature dimensions. (default: 0)')

args = parser.parse_args()

params = {}

for key in vars(args).keys():
    params[key] = vars(args)[key] 

if os.path.exists(args.json):
    js = json_load(args.json)
    for key in js.keys():
        params[key] = js[key]

device = torch.device("cuda:" + str(params["device"])) if torch.cuda.is_available() else torch.device("cpu")

# dataset settings
tudataset = TUDataset(root='./dataset/',name = params["dataset"],use_node_attr=params["use_node_attr"])
dataset_info = info_collect_graph_dataset(tudataset)

# evaluation settings
random_state = range(params["random_state"], params["random_state"] + params["repetitions"])
accs = []

# pytorch settings
solver = 'Adam'
torch.manual_seed(1)

# algorithm settings
if params["num_basis"] == 0:
    params["num_basis"] = dataset_info["max_num_nodes"]
if params["ml_hiddens"] == 0:
    params["ml_hiddens"] = dataset_info["num_features"] 
if params["n_projections"] == 0:
    params["n_projections"] = dataset_info["num_features"]
if params["r"] == 0:
    params["r"] = dataset_info["num_features"]
if params["prep_scheme"] == "GINConv":
    gnn_model = GINConv_no_nn_multi(params["prep_layers"], device, 1.0)
elif params["prep_scheme"] == "WWL_conti":
    gnn_model = GNN_WWL(params["prep_layers"]).to(device)
else:
    gnn_model = GNN_Null().to(device)

# result recording
size_gram_matrix = [dataset_info["num_graphs"], dataset_info["num_graphs"]]
size_seq_gram_matrix = [params["epochs"]] + size_gram_matrix
size_basis_matrix = [params["num_basis"], dataset_info["num_features"] * (params["prep_layers"] + 1)]
size_seq_basis_matrix = [params["epochs"]] + size_basis_matrix

params["method"] = params["baseline"]
if_recorded, folder_name, exp_name  = init_other_result_folders(params, check_only= True)

if if_recorded:
    bmat = np.load(folder_name + "/data/grams_0_0.npy")

full_losses = {}

for idx, method_name in enumerate(params["methods"]):
    if method_name in ["SOT_EMD", "SOT_Sinkhorn"]:
        params["method"] = "SOT"
    elif method_name in ["SOT_ML_EMD", "SOT_ML_Sinkhorn"]:
        params["method"] = "SOT_ML"
    elif method_name in ["SOT_k_EMD", "SOT_k_Sinkhorn"]:
        params["method"] = "SOT_k"
    else:
        params["method"] = method_name

    if method_name in ["SOT_EMD", "SOT_Sinkhorn", "SOT_ML_EMD", "SOT_ML_Sinkhorn"]:
        if_recorded, folder_name, exp_name  = init_SOT_result_folders(params, check_only = True)
    else:
        if_recorded, folder_name, exp_name  = init_other_result_folders(params, check_only= True)
    
    if if_recorded:
        if method_name == "SOT_EMD":
            gmat = np.load(folder_name + "/data/SOT_EMD_grams_0_0.npy")
            err = np.sqrt(np.mean(np.power(gmat - bmat,2), axis=(1,2)))
            full_losses["ASOT-DL"] = [params["colors"][idx], "-", err.tolist()]
        elif method_name == "SOT_Sinkhorn":
            gmat = np.load(folder_name + "/data/eSOT_grams_0_0.npy")
            err = np.sqrt(np.mean(np.power(gmat - bmat,2), axis=(1,2)))
            full_losses["eASOT-DL"] = [params["colors"][idx], "-", err.tolist()]
        elif method_name == "SOT_ML_EMD":
            gmat = np.load(folder_name + "/data/SOT_EMD_grams_0_0.npy")
            err = np.sqrt(np.mean(np.power(gmat - bmat,2), axis=(1,2)))
            full_losses["ASOT-ML"] = [params["colors"][idx], "-", err.tolist()]
        elif method_name == "SOT_ML_Sinkhorn":
            gmat = np.load(folder_name + "/data/eSOT_grams_0_0.npy")
            err = np.sqrt(np.mean(np.power(gmat - bmat,2), axis=(1,2)))
            full_losses["eASOT-ML"] = [params["colors"][idx], "-", err.tolist()]
        elif method_name == "SOT_k_EMD":
            gmat = np.load(folder_name + "/data/SOT_EMD_grams_0_0.npy")
            err = np.sqrt(np.mean(np.power(gmat - bmat,2)))
            full_losses["ASOT-k"] = [params["colors"][idx], "dotted", err.tolist()]
        elif method_name == "SOT_k_Sinkhorn":
            gmat = np.load(folder_name + "/data/eSOT_grams_0_0.npy")
            err = np.sqrt(np.mean(np.power(gmat - bmat,2)))
            full_losses["eASOT-k"] = [params["colors"][idx], "dotted", err.tolist()]
        else:
            gmat = np.load(folder_name + "/data/grams_0_0.npy")
            err = np.sqrt(np.mean(np.power(gmat - bmat,2)))
            full_losses[params["method"]] = [params["colors"][idx], "dotted", err.tolist()]
            
draw_curves_err(params["epochs"], full_losses)
#plt.savefig("fig.pdf")
plt.show()