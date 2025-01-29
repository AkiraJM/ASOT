# Anchor Space Optimal Transport (ASOT)
This repository contains the source codes of the paper [_Anchor Space Optimal Transport as a Fast Solution to Multiple Optimal Transport Problems_](https://ieeexplore.ieee.org/document/10704726), whcih is currently published in IEEE Transactions on Neural Networks and Learning Systems.

As decribed in our paper, we implement our proposed anchor space learning models such as ASOT-ML, ASOT-DL, and ASOT-k, where ASOT-ML is based on a metric leanring framework detailed in Figure 2(a) of our preprint, ASOT-DL is based on a dictionary learning framework (Figure 2(b)), and ASOT-k is another variant based on the mini-batch k-means.

It should be noted that, because we use GPU parallelization, CUDA and cuDNN are required for faster computation. Although these codes are also runnable in a cpu-only environment, we do not recommand to run them on machines without GPU. In details, we run them in an environment with CUDA v11.3+cuDNN 8.2.1.


## Dependencies
The code is successfully tested by using Python 3.7 on Windows 11. It relies on the following dependencies:

- numpy (1.21.6)
- scikit-learn (1.0.2)
- POT (0.9.0)
- torch (1.11.0+cu113)
- torch-cluster (1.6.0)
- torch-geometric (2.0.4)
- torch-scatter (2.0.9)
- torch-sparse (0.6.14)
- torch-spline-conv (1.2.1)
- matplotlib (3.5.2)

## Usage
Script 'exp_SOT_graph_dataset_curves.py' is for distance approximation experiments on graph datasets. The detailed usage document is presented as follows.
It should be noted that, in older versions, our methods are also called 'SOT', which sometimes can be seen from the comments or captions.

```
usage: exp_SOT_graph_dataset_curves.py [-h] [--json JSON] [--device DEVICE]
                                       [--dataset DATASET]
                                       [--use_node_attr USE_NODE_ATTR]
                                       [--n_folds N_FOLDS]
                                       [--random_state RANDOM_STATE]
                                       [--repetitions REPETITIONS]
                                       [--if_single_fold IF_SINGLE_FOLD]
                                       [--lr LR] [--batch_size BATCH_SIZE]
                                       [--epochs EPOCHS]
                                       [--num_basis NUM_BASIS]
                                       [--ml_hiddens ML_HIDDENS]
                                       [--OT_reg OT_REG] [--OT_iter OT_ITER]
                                       [--norm NORM] [--rc RC] [--lp LP]
                                       [--sp SP] [--ml ML]
                                       [--lasso_iters LASSO_ITERS]
                                       [--prep_scheme {GINConv,WWL_conti}]
                                       [--prep_layers PREP_LAYERS]
                                       [--method {SOT,SOT_ML,SOT_k,Sinkhorn,EMD,sliced_Wass}]
                                       [--n_projections N_PROJECTIONS] [--r R]

Experimental code for anchor space optimal transport on tudataset.

optional arguments:
  -h, --help            show this help message and exit
  --json JSON           path of json file for parameter setting. (default:
                        params.json)
  --device DEVICE       which gpu to use if any. (default: 0)
  --dataset DATASET     name of dataset. (default: MUTAG)
  --use_node_attr USE_NODE_ATTR
                        if use node attributes. (default: False)
  --n_folds N_FOLDS     number of folds for dataset splitting. (default: 10)
  --random_state RANDOM_STATE
                        random_state. (default: 0)
  --repetitions REPETITIONS
                        number of repetitions. (default: 1)
  --if_single_fold IF_SINGLE_FOLD
                        if only run a single fold. (default: True)
  --lr LR               learning rate for neural network training. (default:
                        0.01)
  --batch_size BATCH_SIZE
                        training batch size. If 0, use the dataset size.
                        (default: 0)
  --epochs EPOCHS       number of epochs. (default: 500)
  --num_basis NUM_BASIS
                        number of basis vectors of SOT. if 0, use the maximum
                        number of nodes. (default: 0)
  --ml_hiddens ML_HIDDENS
                        number of dimensions of the hidden layer of metric
                        learning variant. if 0, use the number of feature
                        dimensions. (default: 0)
  --OT_reg OT_REG       weight of regularization term of eOT. (default: 0.1)
  --OT_iter OT_ITER     number of iterations of the Sinkhorn's algorithm.
                        (default: 50)
  --norm NORM           order of norm for computing the Minkowski distance.
                        (default: 2)
  --rc RC               weight of the reconstruction loss. (default: 1.0)
  --lp LP               weight of the l_p ball loss. (default: 1.0)
  --sp SP               weight of the simplex constraint violation loss.
                        (default: 1.0)
  --ml ML               weight of the metric learning loss. (default: 10.0)
  --lasso_iters LASSO_ITERS
                        number of the sparse coding layers. (default: 20)
  --prep_scheme {GINConv,WWL_conti}
                        scheme of the node feature preprocessing. (default:
                        GINConv)
  --prep_layers PREP_LAYERS
                        number of layers of the preprocessing scheme.
                        (default: 4)
  --method {SOT,SOT_ML,SOT_k,Sinkhorn,EMD,sliced_Wass}
                        method name. (default: SOT)
  --n_projections N_PROJECTIONS
                        number of projections of the sliced Wasserstein
                        distance. If 0, use the number of feature dimensions.
                        (default: 0)
  --r R                 number of ranks of the low rank Sinkhorn. If 0, use
                        the number of feature dimensions. (default: 0)
```

Script 'exp_draw_curves.py' is for result visualization, which presents the curves of RMSE. The detailed usage document is presented as follows. To show the same result as the one computed by 'exp_SOT_graph_dataset_curves.py', you show check if the parameters are also the same.

```
usage: exp_draw_curves.py [-h] [--json JSON] [--exp_type EXP_TYPE]
                          [--device DEVICE] [--dataset DATASET]
                          [--use_node_attr USE_NODE_ATTR] [--n_folds N_FOLDS]
                          [--random_state RANDOM_STATE]
                          [--repetitions REPETITIONS]
                          [--if_single_fold IF_SINGLE_FOLD] [--lr LR]
                          [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                          [--num_basis NUM_BASIS] [--ml_hiddens ML_HIDDENS]
                          [--OT_reg OT_REG] [--OT_iter OT_ITER] [--norm NORM]
                          [--rc RC] [--lp LP] [--sp SP] [--ml ML]
                          [--lasso_iters LASSO_ITERS]
                          [--prep_scheme {GINConv,WWL_conti}]
                          [--prep_layers PREP_LAYERS]
                          [--baseline {EMD,Sinkhorn}]
                          [--methods METHODS [METHODS ...]]
                          [--colors COLORS [COLORS ...]]
                          [--n_projections N_PROJECTIONS] [--r R]

Experimental code for anchor space optimal transport on tudataset.

optional arguments:
  -h, --help            show this help message and exit
  --json JSON           path of json file for parameter setting. (default:
                        params.json)
  --exp_type EXP_TYPE   type of experiment. (default: graph)
  --device DEVICE       which gpu to use if any. (default: 0)
  --dataset DATASET     name of dataset. (default: MUTAG)
  --use_node_attr USE_NODE_ATTR
                        if use node attributes. (default: False)
  --n_folds N_FOLDS     number of folds for dataset splitting. (default: 10)
  --random_state RANDOM_STATE
                        random_state. (default: 0)
  --repetitions REPETITIONS
                        number of repetitions. (default: 1)
  --if_single_fold IF_SINGLE_FOLD
                        if only run a single fold. (default: True)
  --lr LR               learning rate for neural network training. (default:
                        0.01)
  --batch_size BATCH_SIZE
                        training batch size. If 0, use the dataset size.
                        (default: 0)
  --epochs EPOCHS       number of epochs. (default: 500)
  --num_basis NUM_BASIS
                        number of basis vectors of SOT. if 0, use the maximum
                        number of nodes. (default: 0)
  --ml_hiddens ML_HIDDENS
                        number of dimensions of the hidden layer of metric
                        learning variant. if 0, use the number of feature
                        dimensions. (default: 0)
  --OT_reg OT_REG       weight of regularization term of eOT. (default: 0.1)
  --OT_iter OT_ITER     number of iterations of the Sinkhorn's algorithm.
                        (default: 0.1)
  --norm NORM           order of norm for computing the Minkowski distance.
                        (default: 2)
  --rc RC               weight of the reconstruction loss. (default: 1.0)
  --lp LP               weight of the l_p ball loss. (default: 1.0)
  --sp SP               weight of the simplex constraint violation loss.
                        (default: 1.0)
  --ml ML               weight of the metric learning loss. (default: 10.0)
  --lasso_iters LASSO_ITERS
                        number of the sparse coding layers. (default: 20)
  --prep_scheme {GINConv,WWL_conti}
                        scheme of the node feature preprocessing. (default:
                        GINConv)
  --prep_layers PREP_LAYERS
                        number of layers of the preprocessing scheme.
                        (default: 4)
  --baseline {EMD,Sinkhorn}
                        baseline method name. (default: EMD)
  --methods METHODS [METHODS ...]
                        names of methods. (default: SOT_EMD, SOT_Sinkhorn,
                        SOT_ML_EMD, SOT_ML_Sinkhorn, SOT_k_EMD,
                        SOT_k_Sinkhorn, Sinkhorn, sliced_Wass)
  --colors COLORS [COLORS ...]
                        colors of curves. (default: red, orange, blue,
                        deepskyblue, purple, magenta, green, lime)
  --n_projections N_PROJECTIONS
                        number of projections of the sliced Wasserstein
                        distance. If 0, use the number of feature dimensions.
                        (default: 0)
  --r R                 number of ranks of the low rank Sinkhorn. If 0, use
                        the number of feature dimensions. (default: 0)
```

## Run the experiment
In the simplest instance, you can directly run the experiment script 'example.bat'. For machines with Linux or MacOS, you should copy the commands in the script, and then run them on your terminal. The commands are presented below.

```bash
python exp_SOT_graph_dataset_curves.py --dataset MUTAG --method EMD
python exp_SOT_graph_dataset_curves.py --dataset MUTAG --method Sinkhorn
python exp_SOT_graph_dataset_curves.py --dataset MUTAG --method SOT --epochs 2000
python exp_SOT_graph_dataset_curves.py --dataset MUTAG --method SOT_ML --epochs 2000 --ml 100.0
python exp_SOT_graph_dataset_curves.py --dataset MUTAG --method SOT_k
python exp_SOT_graph_dataset_curves.py --dataset MUTAG --method sliced_Wass

python exp_draw_curves.py --dataset MUTAG --epochs 2000 --ml 100.0
```
which output the curves of approximation errors of the MUTAG dataset (Figure 3 of our preprint).

For the time costs, they are recorded in a json file 'result.json'. For an example, after running the commands above, you can find the detailed records in './results/MUTAG/exp_SOT_bd350f2d483338d390409f219fbc3e10f8d6ce63/result.json', which is a python Dict data. In this json file, "records" -> "time" -> "time_eSOT_test", "time_SOT_train", "time_SOT_EMD_test" show the time costs of eSOT testing, model training, and EMD testing, which are all lists of sizes equal to the number of epochs.

It should be noted that, because we compute distances with both Sinkhorn and EMD solvers for each epoch, where the time cost of EMD solver is usually very high, it might take a time until finishing all experiments.

## Citation
If you make use of our code, please use the following Bibtex citation.
```
@article{huang2024anchor,
  title={Anchor Space Optimal Transport as a Fast Solution to Multiple Optimal Transport Problems},
  author={Huang, Jianming and Su, Xun and Fang, Zhongxi and Kasai, Hiroyuki},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```