import os
import numpy as np
import torch
import hashlib
import json

# params

def sot_params_keylist(variant = "SOT"):
    if variant == "SOT":
        return ["lr", "batch_size", "epochs", "num_basis", "OT_reg", "OT_iter", "norm", "rc", "lp", "sp", "lasso_iters"]
    elif variant == "SOT_k":
        return ["num_basis", "OT_reg", "OT_iter", "norm"]
    elif variant == "SOT_EMD":
        return ["lr", "batch_size", "epochs", "num_basis", "norm", "rc", "lp", "sp", "lasso_iters"]
    elif variant == "SOT_k_EMD":
        return ["num_basis", "norm"]
    elif variant == "SOT_ML":
        return ["lr", "batch_size", "epochs", "num_basis", "ml_hiddens", "OT_reg", "OT_iter", "norm", "ml"]
    else:
        return []

def other_methods_param_keylist(alg =  "EMD"):
    if alg == "Sinkhorn":
        return ["OT_reg", "OT_iter", "norm"]
    elif alg == "sliced_Wass":
        return ["n_projections"]
    elif alg == "low_rank_Sinkhorn":
        return ["r"]
    elif alg ==  "SOT_k":
        return ["num_basis", "OT_reg", "OT_iter", "norm"]
    else:
        return []

def exp_params_keylist(params):
    if params["exp_type"] == "graph":
        return ["method", "dataset", "use_node_attr", "n_folds", "random_state", "repetitions", "if_single_fold", "prep_scheme", "prep_layers"]
    elif params["exp_type"] == "gaussian":
        return ["method", "num_distributions", "num_dimensions", "num_max_pts_each_distrib", "num_min_pts_each_distrib", "random_state", "repetitions", "if_single_fold"]
    else:
        return ["method", "dataset", "random_state", "repetitions", "if_single_fold"]

def get_sub_dict(dict, keylist):
    subdict = {}
    for key in keylist:
        subdict[key] = dict[key]
    return subdict

def params_to_filename(params, keylist = []):
    file_name = ""
    if len(keylist) <= 0:
        for key, item in params.items():
            file_name += '_' if len(file_name) > 0 else ''
            file_name += key + '=' + str(item)
    else:
        for key in keylist:
            file_name += '_' if len(file_name) > 0 else ''
            file_name += key + '=' + str(params[key])
    return file_name

# result recording

def json_load(filename):
    with open(filename, 'r') as f:
        ret = json.load(f)
    f.close()
    return ret

def json_write(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
    f.close()

def record_reference(folder_name, exp_name, params):
    refer_dict_name = "./results/ref.json"
    dataset_name = params["dataset"]
    if not os.path.exists(refer_dict_name):
        refer_dict = {}
    else:
        refer_dict = json_load(refer_dict_name)
    if not (dataset_name in refer_dict.keys()):
        refer_dict[dataset_name] = {}
    refer_dict[dataset_name][folder_name] = exp_name
    json_write(refer_dict, refer_dict_name)

def init_other_result_folders(params, check_only = False):
    exp_name = params_to_filename(params, keylist = exp_params_keylist(params) + other_methods_param_keylist(alg = params["method"]))
    folder_name = "./results/" + params["dataset"] + '/exp_'+params["method"] + '_' + hashlib.sha1(exp_name.encode("utf-8")).hexdigest()
    if not os.path.exists(folder_name):
        if not check_only:
            os.makedirs(folder_name)
            os.mkdir(folder_name + "/data")
        return False, folder_name, exp_name
    else:
        return True, folder_name, exp_name

def init_SOT_result_folders(params, check_only = False):
    exp_name = params_to_filename(params, keylist = exp_params_keylist(params) + sot_params_keylist(variant = params["method"]))
    folder_name = "./results/" + params["dataset"] + '/exp_' + params["method"] + '_' + hashlib.sha1(exp_name.encode("utf-8")).hexdigest()
    if not os.path.exists(folder_name):
        if not check_only:
            os.makedirs(folder_name)
            os.mkdir(folder_name + "/data")
            os.mkdir(folder_name + "/ckpts")
        return False, folder_name, exp_name
    else:
        return True, folder_name, exp_name

def init_other_result_dict(params):
    res = {
        "method":params["method"],
        "method_params":get_sub_dict(params, other_methods_param_keylist(alg = params["method"])),
        "exp_params":get_sub_dict(params, exp_params_keylist(params)),
        "status":{
            "cur_rep":0,
            "cur_fold":0,
        },
        "records":[],
    }

    for rep in range(params["repetitions"]):
        res["records"].append([])
        for fold in range(params["n_folds"]):
            res["records"][rep].append({"err":0.,
                                        "time":0.})

    return res

def init_SOT_result_dict(params):
    res = {
        "method":"SOT",
        "sot_params":get_sub_dict(params, sot_params_keylist(params["method"])),
        "exp_params":get_sub_dict(params, exp_params_keylist(params)),
        "status":{
            "cur_rep":0,
            "cur_fold":0,
            "cur_epoch":0,
        },
        "records":[],
    }

    for rep in range(params["repetitions"]):
        res["records"].append([])
        for fold in range(params["n_folds"]):
            res["records"][rep].append({ "err":{
            "err_eSOT_Sinkhorn":[],
            "err_SOT_EMD":[],
            # "err_eSOT_k_Sinkhorn":0.,
            # "err_SOT_k_EMD":0.,
        },
        "loss_curve":{
            "train_loss_rc":[],
            "train_loss_lp":[],
            "train_loss_sp":[],
        },
        "time":{
            "time_eSOT_test":[],
            # "time_eSOT_k_test":[],
            "time_SOT_train":[],
            # "time_kmeans_train":0.,
            "time_SOT_EMD_test":[],
            # "time_SOT_k_EMD_test":0.,
        },
        # "data":{
        #     "file_eSOT_grams":"",
        #     # "file_eSOT_k_gram":"",
        #     "file_SOT_EMD_grams":"",
        #     # "file_SOT_k_EMD_gram":"",
        #     "file_SOT_basis":"",
        #     # "file_SOT_k_basis":"",
        #     "file_cur_model_ckpt":"",
        #     "file_best_model_ckpt":"",}
        })

    return res

# file I/O

def Check_and_Load(file_names):
    ifexists = []
    loaded = []
    for file_name in file_names:
        ok = os.path.exists(file_name + '.npy')
        ifexists.append(ok)
        if ok:
            loaded.append(np.load(file_name + '.npy'))
        else:
            loaded.append(None)
    return ifexists,loaded

def Save_Mat(file_name, matrix):
    np.save(file_name + '.npy', matrix)

def Save_Mats(file_names, matrices):
    for file_name, matrix in zip(file_names, matrices):
        np.save(file_name + '.npy', matrix)