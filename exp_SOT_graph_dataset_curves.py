import sys
import os
import argparse
import copy
from tabnanny import verbose

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
from utilities.SOT import SubspaceOT, SubspaceOT_kmeans, SubspaceOT_ML, SubspaceOT_ML_ex, pairwise_subspaceOT_distance, pairwise_wasserstein_distance_sinkhorn, asym_pairwise_subspaceOT_distance, pairwise_Minkowski_distance
from utilities.OTmethods import pairwise_wasserstein_distance, asym_pairwise_wasserstein_distance, pairwise_SOT_distance

# arguments
parser = argparse.ArgumentParser(description='Experimental code for anchor space optimal transport on tudataset.')

parser.add_argument('--json', type=str, default="params.json",
                        help='path of json file for parameter setting. (default: params.json)')
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
                        help='number of iterations of the Sinkhorn\'s algorithm. (default: 50)')
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
parser.add_argument('--method', type=str, default="SOT", choices=["SOT", "SOT_ML", "SOT_k", "Sinkhorn", "EMD", "sliced_Wass"],
                        help='method name. (default: SOT)')
parser.add_argument('--n_projections', type=int, default=0,
                        help='number of projections of the sliced Wasserstein distance. If 0, use the number of feature dimensions. (default: 0)')
parser.add_argument('--r', type=int, default=0,
                        help='number of ranks of the low rank Sinkhorn. If 0, use the number of feature dimensions. (default: 0)')

args = parser.parse_args()

params = {"exp_type" : "graph"}

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

if params["method"] == "SOT" or params["method"] == "SOT_ML":
    if_recorded, folder_name, exp_name  = init_SOT_result_folders(params)
else:
    if_recorded, folder_name, exp_name  = init_other_result_folders(params)

if if_recorded:
    resrec = json_load(folder_name + '/result.json')
elif params["method"] == "SOT" or params["method"] == "SOT_ML":
    resrec = init_SOT_result_dict(params)
    json_write(resrec, folder_name + '/result.json')
else:
    resrec = init_other_result_dict(params)
    json_write(resrec, folder_name + '/result.json')


record_reference(folder_name, exp_name, params)

# train funcs
def SOT_ML_precompute_distance(train_loader):
    list_prep_D = []

    for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            x = gnn_model(data.x, data.edge_index)
            list_prep_D.append(pairwise_Minkowski_distance(x))
    
    return list_prep_D

def SOT_ML_train_one_epoch(ep, model, train_loader, list_prep_D, record):
    model.train()

    total_loss_lp = 0
    total_loss_rc = 0
    total_loss_sp = 0
    total_loss_ml = 0

    if not "train_loss_ml" in record["loss_curve"].keys():
        record["loss_curve"]["train_loss_ml"] = []

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        x = gnn_model(data.x, data.edge_index)

        time_encoding = time.time()
        x_c, x_p = model(x)
        loss_lp = model.loss_lp(x_c)
        loss_rc = model.loss_rc(x, x_p)
        loss_sp = model.loss_sp(x_c)
        loss_ml = model.loss_ml(list_prep_D[batch_idx], x_c)
        final_loss = params["ml"] * loss_ml
        time_encoding = time.time() - time_encoding

        total_loss_lp += loss_lp
        total_loss_rc += loss_rc
        total_loss_sp += loss_sp
        total_loss_ml += loss_ml
        
        time_gradient = time.time()
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        time_gradient = time.time() - time_gradient
        if len(record["time"]["time_SOT_train"]) == ep:
            record["time"]["time_SOT_train"].append(time_encoding + time_gradient)
        else:
            record["time"]["time_SOT_train"][ep] = time_encoding + time_gradient
    
    total_loss_lp /= len(train_loader)
    total_loss_rc /= len(train_loader)
    total_loss_sp /= len(train_loader)
    total_loss_ml /= len(train_loader)

    if len(record["loss_curve"]["train_loss_lp"]) == ep:
        record["loss_curve"]["train_loss_lp"].append(total_loss_lp.item())
    else:
        record["loss_curve"]["train_loss_lp"][ep] = total_loss_lp.item()

    if len(record["loss_curve"]["train_loss_rc"]) == ep:
        record["loss_curve"]["train_loss_rc"].append(total_loss_rc.item())
    else:
        record["loss_curve"]["train_loss_rc"][ep] = total_loss_rc.item()

    if len(record["loss_curve"]["train_loss_sp"]) == ep:
        record["loss_curve"]["train_loss_sp"].append(total_loss_sp.item())
    else:
        record["loss_curve"]["train_loss_sp"][ep] = total_loss_sp.item()

    if len(record["loss_curve"]["train_loss_ml"]) == ep:
        record["loss_curve"]["train_loss_ml"].append(total_loss_ml.item())
    else:
        record["loss_curve"]["train_loss_ml"][ep] = total_loss_ml.item()
    
    return model.get_basis().cpu().detach().numpy(), total_loss_lp.item(), total_loss_rc.item(), total_loss_sp.item(), total_loss_ml.item()

def SOT_train_one_epoch(ep, model, train_loader, record):

    model.train()

    total_loss_lp = 0
    total_loss_rc = 0
    total_loss_sp = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        x = gnn_model(data.x, data.edge_index)

        time_encoding = time.time()
        x_c, x_p = model(x)
        loss_lp = model.loss_lp(x_c)
        loss_rc = model.loss_rc(x, x_p)
        loss_sp = model.loss_sp(x_c)
        final_loss = params["lp"] * loss_lp + params["rc"] * loss_rc + params["sp"] * loss_sp
        time_encoding = time.time() - time_encoding

        total_loss_lp += loss_lp
        total_loss_rc += loss_rc
        total_loss_sp += loss_sp
        
        time_gradient = time.time()
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        time_gradient = time.time() - time_gradient
        if len(record["time"]["time_SOT_train"]) == ep:
            record["time"]["time_SOT_train"].append(time_encoding + time_gradient)
        else:
            record["time"]["time_SOT_train"][ep] = time_encoding + time_gradient
    
    total_loss_lp /= len(train_loader)
    total_loss_rc /= len(train_loader)
    total_loss_sp /= len(train_loader)

    if len(record["loss_curve"]["train_loss_lp"]) == ep:
        record["loss_curve"]["train_loss_lp"].append(total_loss_lp.item())
    else:
        record["loss_curve"]["train_loss_lp"][ep] = total_loss_lp.item()

    if len(record["loss_curve"]["train_loss_rc"]) == ep:
        record["loss_curve"]["train_loss_rc"].append(total_loss_rc.item())
    else:
        record["loss_curve"]["train_loss_rc"][ep] = total_loss_rc.item()

    if len(record["loss_curve"]["train_loss_sp"]) == ep:
        record["loss_curve"]["train_loss_sp"].append(total_loss_sp.item())
    else:
        record["loss_curve"]["train_loss_sp"][ep] = total_loss_sp.item()
    
    return model.get_basis().cpu().detach().numpy(), total_loss_lp.item(), total_loss_rc.item(), total_loss_sp.item()

def SOT_kmeans_train(model_kmeans, train_loader, record):
    if not isinstance(record["time"], Dict):
        record["time"] = {}

    record["time"]["time_train"] = 0.

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        x = gnn_model(data.x, data.edge_index)

        time_sot_kmeans = time.time()
        model_kmeans.partial_fit(x)
        time_sot_kmeans = time.time() - time_sot_kmeans

        record["time"]["time_train"] += time_sot_kmeans

    record["inertia"] = model_kmeans.inertia()

def SOT_kmeans_eval(model_kmeans, full_dataset_loader, record):
    if not isinstance(record["time"], Dict):
        record["time"] = {}

    record["time"]["time_test_Sinkhorn"] = 0.
    record["time"]["time_test_EMD"] = 0.

    for batch_idx, data in enumerate(full_dataset_loader):
        
        data = data.to(device)

        test_x = gnn_model(data.x, data.edge_index)
        test_batch = data.batch

        time_sot_kmeans = time.time()
        eSOT_k_mat = pairwise_subspaceOT_distance(model_kmeans, test_x, test_batch, p = params["norm"], reg = params["OT_reg"], iters = params["OT_iter"])
        time_sot_kmeans = time.time() - time_sot_kmeans

        record["time"]["time_test_Sinkhorn"] = time_sot_kmeans

        time_sot_kmeans = time.time()
        x_c_k, x_p_k = model_kmeans(test_x)
        mass_k = scatter_add(x_c_k, test_batch, dim = 0)
        SOT_emd_mat = pairwise_SOT_distance(model_kmeans.get_basis(), mass_k, p = params["norm"], reg = params["OT_reg"], maxIter = params["OT_iter"], method = 'emd')
        time_sot_kmeans = time.time() - time_sot_kmeans

        record["time"]["time_test_EMD"] = time_sot_kmeans

    return eSOT_k_mat.cpu().detach().numpy(), SOT_emd_mat, model_kmeans.get_basis().cpu().detach().numpy()

def SOT_eval(ep, model, full_dataset_loader, record):

    model.eval()

    for batch_idx, data in enumerate(full_dataset_loader):
        
        data = data.to(device)

        train_x = gnn_model(data.x, data.edge_index)
        train_batch = data.batch

        time_sot = time.time()
        eSOT_mat = pairwise_subspaceOT_distance(model, train_x, train_batch, p = params["norm"], reg = params["OT_reg"], iters = params["OT_iter"])
        time_sot = time.time() - time_sot
        if len(record["time"]["time_eSOT_test"]) == ep:
            record["time"]["time_eSOT_test"].append(time_sot)
        else:
            record["time"]["time_eSOT_test"][ep] = time_sot

        time_encoding = time.time()
        x_c, x_p = model(train_x)
        mass = scatter_add(x_c, train_batch, dim = 0)
        #mass = torch.nn.functional.normalize(mass, p=1)
        basis_vectors = torch.relu(model.basis.T)
        time_encoding = time.time() - time_encoding

        time_sot = time.time()
        SOT_mat = pairwise_SOT_distance(basis_vectors, mass, cost_matrix=model.subspace_cost_matrix(params["norm"]).cpu().detach().numpy(), p = params["norm"], reg = params["OT_reg"], maxIter = params["OT_iter"], method = 'emd')
        time_sot = time.time() - time_sot
        if len(record["time"]["time_SOT_EMD_test"]) == ep:
            record["time"]["time_SOT_EMD_test"].append(time_sot)
        else:
            record["time"]["time_SOT_EMD_test"][ep] = time_sot

    return eSOT_mat.cpu().detach().numpy(), SOT_mat

def OT_EMD_compute(full_dataset_loader, record):
    for batch_idx, data in enumerate(full_dataset_loader):
        
        data = data.to(device)

        train_x = gnn_model(data.x, data.edge_index)
        train_batch = data.batch

        time_emd = time.time()
        EMD_mat = pairwise_wasserstein_distance(train_x, train_batch, p = params["norm"], reg = params["OT_reg"], maxIter = params["OT_iter"], method = 'emd')
        time_emd = time.time() - time_emd

        record["time"] = time_emd
    
    return EMD_mat

def eOT_Sinkhorn_compute(full_dataset_loader, record):
    for batch_idx, data in enumerate(full_dataset_loader):
        
        data = data.to(device)

        train_x = gnn_model(data.x, data.edge_index)
        train_batch = data.batch

        time_sink = time.time()
        Sink_mat = pairwise_wasserstein_distance_sinkhorn(train_x, train_batch, p = params["norm"], reg = params["OT_reg"], maxIter = params["OT_iter"]).cpu().detach().numpy()
        time_sink = time.time() - time_sink

        record["time"] = time_sink
    
    return Sink_mat

def sliced_Wass_compute(full_dataset_loader, record):
    for batch_idx, data in enumerate(full_dataset_loader):
        
        data = data.to(device)

        test_x = gnn_model(data.x, data.edge_index)
        test_batch = data.batch

        time_sw = time.time()
        SW_mat = pairwise_wasserstein_distance(test_x, test_batch, p = params["norm"], reg = params["OT_reg"], maxIter = params["OT_iter"], method = 'sliced', n_projections = params["n_projections"])
        time_sw = time.time() - time_sw

        record["time"] = time_sw
    
    return SW_mat

def low_rank_Sinkhorn_compute(full_dataset_loader, record):
    for batch_idx, data in enumerate(full_dataset_loader):
        
        data = data.to(device)

        test_x = gnn_model(data.x, data.edge_index)
        test_batch = data.batch

        time_lot = time.time()
        lot_mat = pairwise_wasserstein_distance(test_x, test_batch, p = params["norm"], reg = params["OT_reg"], maxIter = params["OT_iter"], method = 'lr', r = params["r"])
        time_lot = time.time() - time_lot

        record["time"] = time_lot
    
    return lot_mat

# experiment
for ridx,each in enumerate(random_state):
    if resrec["status"]["cur_rep"] > ridx:
        continue

    sfolder = StratifiedKFold(n_splits=params["n_folds"], shuffle=True, random_state=each)
    for fold_idx , (Train_index, Test_index) in enumerate(sfolder.split(dataset_info["G"], dataset_info["y"])):
        if resrec["status"]["cur_fold"] > fold_idx:
            continue

        resrec["status"]["cur_rep"] = ridx
        resrec["status"]["cur_fold"] = fold_idx

        # dataset splitting
        train_dataset = [dataset_info["G"][idx] for idx in Train_index.tolist()]
        test_dataset = [dataset_info["G"][idx] for idx in Test_index.tolist()]

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=params["batch_size"] if params["batch_size"]!=0 else dataset_info["num_graphs"],
                                  shuffle=False)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=params["batch_size"] if params["batch_size"]!=0 else dataset_info["num_graphs"],
                                 shuffle=False)
        full_dataset_loader = DataLoader(dataset=dataset_info["G"],
                                  batch_size=dataset_info["num_graphs"],
                                  shuffle=False)

        if params["method"] == "SOT":
            # model initialization
            model = SubspaceOT(dataset_info["num_features"] * (params["prep_layers"] + 1), params["num_basis"], device, lasso_iters=params["lasso_iters"])

            if solver == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])

            best_score = np.inf

            basis_mat = np.zeros(size_seq_basis_matrix)
            if params["if_single_fold"]:
                eSOT_gram_mat = np.zeros(size_seq_gram_matrix)
                SOT_gram_mat = np.zeros(size_seq_gram_matrix)
            else:
                eSOT_gram_mat = np.zeros(size_gram_matrix)
                SOT_gram_mat = np.zeros(size_gram_matrix)

            for epoch in range(params["epochs"]):
                basis_mat[epoch,:,:], loss_lp, loss_rc, loss_sp  = SOT_train_one_epoch(epoch, model, train_loader, resrec["records"][ridx][fold_idx])

                final_loss = loss_lp * params["lp"] + loss_rc * params["rc"] + loss_sp * params["sp"]
                if best_score > final_loss:
                    best_score = final_loss
                    torch.save(model, folder_name+"/ckpts/best_model_" + str(ridx) + "_" + str(fold_idx) + ".ckpt")
                print('Epoch: %03d/%03d' % (epoch + 1, params["epochs"]))
                print('train_loss_lp: %.4f | train_loss_rc: %.4f | train_loss_sp: %.4f' % (loss_lp, loss_rc, loss_sp))

                if params["if_single_fold"]:
                    eSOT_gram_mat[epoch,:,:], SOT_gram_mat[epoch,:,:] = SOT_eval(epoch, model, full_dataset_loader, resrec["records"][ridx][fold_idx])

            if not params["if_single_fold"]:
                best_model = torch.load(folder_name+"/ckpts/best_model_" + str(ridx) + "_" + str(fold_idx) + ".ckpt")
                eSOT_gram_mat, SOT_gram_mat = SOT_eval(0, best_model, full_dataset_loader, resrec["records"][ridx][fold_idx])

            np.save(folder_name+"/data/SOT_basis_" + str(ridx) + "_" + str(fold_idx) + ".npy", basis_mat)
            np.save(folder_name+"/data/eSOT_grams_" + str(ridx) + "_" + str(fold_idx) + ".npy", eSOT_gram_mat)
            np.save(folder_name+"/data/SOT_EMD_grams_"  + str(ridx) + "_" + str(fold_idx) + ".npy", SOT_gram_mat)
            torch.save(model, folder_name+"/ckpts/final_model_" + str(ridx) + "_" + str(fold_idx) + ".ckpt")
            json_write(resrec, folder_name + '/result.json')
        elif params["method"] == "SOT_ML":
             # model initialization
            model = SubspaceOT_ML_ex(dataset_info["num_features"] * (params["prep_layers"] + 1), params["num_basis"], params["ml_hiddens"], device, lasso_iters=params["lasso_iters"])

            if solver == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])

            best_score = np.inf

            basis_mat = np.zeros(size_seq_basis_matrix)
            if params["if_single_fold"]:
                eSOT_gram_mat = np.zeros(size_seq_gram_matrix)
                SOT_gram_mat = np.zeros(size_seq_gram_matrix)
            else:
                eSOT_gram_mat = np.zeros(size_gram_matrix)
                SOT_gram_mat = np.zeros(size_gram_matrix)

            list_prep_D = SOT_ML_precompute_distance(train_loader)

            for epoch in range(params["epochs"]):
                basis_mat[epoch,:,:], loss_lp, loss_rc, loss_sp, loss_ml  = SOT_ML_train_one_epoch(epoch, model, train_loader, list_prep_D, resrec["records"][ridx][fold_idx])

                final_loss = loss_ml * params["ml"] + loss_rc * params["rc"] + loss_sp * params["sp"]
                if best_score > final_loss:
                    best_score = final_loss
                    torch.save(model, folder_name+"/ckpts/best_model_" + str(ridx) + "_" + str(fold_idx) + ".ckpt")
                print('Epoch: %03d/%03d' % (epoch + 1, params["epochs"]))
                print('train_loss_lp: %.4f | train_loss_rc: %.4f | train_loss_sp: %.4f | train_loss_ml: %.4f' % (loss_lp, loss_rc, loss_sp, loss_ml))

                if params["if_single_fold"]:
                    eSOT_gram_mat[epoch,:,:], SOT_gram_mat[epoch,:,:] = SOT_eval(epoch, model, full_dataset_loader, resrec["records"][ridx][fold_idx])

            if not params["if_single_fold"]:
                best_model = torch.load(folder_name+"/ckpts/best_model_" + str(ridx) + "_" + str(fold_idx) + ".ckpt")
                eSOT_gram_mat, SOT_gram_mat = SOT_eval(0, best_model, full_dataset_loader, resrec["records"][ridx][fold_idx])

            np.save(folder_name+"/data/SOT_basis_" + str(ridx) + "_" + str(fold_idx) + ".npy", basis_mat)
            np.save(folder_name+"/data/eSOT_grams_" + str(ridx) + "_" + str(fold_idx) + ".npy", eSOT_gram_mat)
            np.save(folder_name+"/data/SOT_EMD_grams_"  + str(ridx) + "_" + str(fold_idx) + ".npy", SOT_gram_mat)
            torch.save(model, folder_name+"/ckpts/final_model_" + str(ridx) + "_" + str(fold_idx) + ".ckpt")
            json_write(resrec, folder_name + '/result.json')
        elif params["method"] == "SOT_k":
            model = SubspaceOT_kmeans(2048, dataset_info["num_features"] * (params["prep_layers"] + 1), params["num_basis"], device, lasso_iters=params["lasso_iters"])

            SOT_kmeans_train(model, train_loader, resrec["records"][ridx][fold_idx])

            eSOT_gram_mat, SOT_gram_mat, basis_mat = SOT_kmeans_eval(model, full_dataset_loader, resrec["records"][ridx][fold_idx])

            np.save(folder_name+"/data/SOT_basis_" + str(ridx) + "_" + str(fold_idx) + ".npy", basis_mat)
            np.save(folder_name+"/data/eSOT_grams_" + str(ridx) + "_" + str(fold_idx) + ".npy", eSOT_gram_mat)
            np.save(folder_name+"/data/SOT_EMD_grams_"  + str(ridx) + "_" + str(fold_idx) + ".npy", SOT_gram_mat)
            json_write(resrec, folder_name + '/result.json')
        elif params["method"] == "Sinkhorn":
            Sink_mat = eOT_Sinkhorn_compute(full_dataset_loader, resrec["records"][ridx][fold_idx])
            np.save(folder_name+"/data/grams_" + str(ridx) + "_" + str(fold_idx) + ".npy",Sink_mat)
            print("computational time: %.4f" % (resrec["records"][ridx][fold_idx]["time"]))
            json_write(resrec, folder_name + '/result.json')
        elif params["method"] == "EMD":
            EMD_mat = OT_EMD_compute(full_dataset_loader, resrec["records"][ridx][fold_idx])
            np.save(folder_name+"/data/grams_" + str(ridx) + "_" + str(fold_idx) + ".npy",EMD_mat)
            print("computational time: %.4f" % (resrec["records"][ridx][fold_idx]["time"]))
            json_write(resrec, folder_name + '/result.json')
        elif params["method"] == "sliced_Wass":
            SW_mat = sliced_Wass_compute(full_dataset_loader, resrec["records"][ridx][fold_idx])
            np.save(folder_name+"/data/grams_" + str(ridx) + "_" + str(fold_idx) + ".npy",SW_mat)
            print("computational time: %.4f" % (resrec["records"][ridx][fold_idx]["time"]))
            json_write(resrec, folder_name + '/result.json')
        elif params["method"] == "low_rank_Sinkhorn":
            lot_mat = low_rank_Sinkhorn_compute(full_dataset_loader, resrec["records"][ridx][fold_idx])
            np.save(folder_name+"/data/grams_" + str(ridx) + "_" + str(fold_idx) + ".npy",lot_mat)
            print("computational time: %.4f" % (resrec["records"][ridx][fold_idx]["time"]))
            json_write(resrec, folder_name + '/result.json')

        if params["if_single_fold"]:
            break
           