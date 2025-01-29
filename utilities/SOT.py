from typing import Callable, Optional, Union, Tuple

import torch
import math
import time

from torch import Tensor
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, mul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch.nn.modules.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import degree

from sklearn.cluster import MiniBatchKMeans

# Function utilities

def split(batch : torch.LongTensor):
    r"""
    Batch split function. This function takes an index vector as the input. It returns a list of numbers of elements of each index.

    Args:
        batch (LongTensor) - The index vector of shape (n,1), where n denotes the number of elements. Each row of it denots the index of an elements.

    Rtype:
        sum (list[int]) - The list of numbers of elements of each index.
    """

    n = batch.shape[0]
    y = torch.ones((n,1))
    return scatter_add(y, batch.cpu(), dim = 0).cpu().detach().numpy().astype(int).reshape(-1)

def minmax_normalize(X):
    r"""
    Min-max normalization function, which normalizes a vecotr/matrix with the following formula.
    
    x' = ( x - min(x) ) / ( max(x) - min(x) )

    Args:
        x (Tensor)

    Rtype
        x' (Tensor)
    """

    X_min = X.min()
    X_max = X.max()
    X = (X - X_min)/(X_max - X_min)
    return X

def pairwise_blocklize(indices : torch.LongTensor, func, block_size : int, **kwargs):
    r"""
    Computation blocklizing function, which blocklize the computation process in case of an overflow of memory. It requires an input of a predefined function that handles a block of data, whose indices are stored in an index vector/matrix.

    Args:
        indices (LongTensor) - The index vector/matrix of shape (n,k), where n denotes the number of the parameter sets of a single computation process, k denotes the number of parameters of each of them.
        func (Function) - The predefined processing function, it takes indices[b,:] as one of its inputs, where b denotes the number of parameter sets in a single block.
        block_size (int) - The size of each block.
        kwargs - Other parameters that will be inputted into func.

    Rtype
        res (list) - The list of outputs of every blocks.
    """

    num_total = len(indices)
    num_blocks = int(math.floor(num_total / block_size))
    res = []
    for i in range(num_blocks + 1):
        if i*block_size >= num_total:
            break
        ret = func(indices[ i*block_size : num_total if (i+1)*block_size > num_total else (i+1)*block_size, :], **kwargs)
        res.append(ret)
    return res

def sinkhorn(a : torch.Tensor, b : torch.Tensor, C : torch.Tensor, reg : float = 1.0, iters : int = 50):
    r"""
    The function of the Sinkhorn's algorithm for computing 1-Wasserstein distance implemented with torch.

    Args:
        a (Tensor) - The first mass vector of shape (n,1), where n denotes the size of the first discrete distribution.
        b (Tensor) - The second mass vector of shape (m,1), where m denotest the size of the second discrete distribution.
        C (Tensor) - The ground cost matrix of shape (n,m). Each element of it denotes the distance of a pair of points from the two distributions.
        reg (float) - The weight of regularization term (epsilon). (default: 1.0)
        iters (int) - The number of the iterations. (default: 50)

    Rtype:
        distance (Tensor) - The 1-Wasserstein distance.
    """

    u = a.reshape(-1,1)
    v = b.reshape(-1,1)
    N = a.shape[0]
    M = b.shape[0]
    P = torch.zeros(N,M).to(C.device)
    K = torch.exp(-(C / reg))

    for it in range(iters):
        u = a / (K @ v)
        v = b / (K.T @ u)

    P = torch.diag_embed(u.view(-1)) @ K @ torch.diag_embed(v.view(-1))

    return torch.sum(P * C)

def sinkhorn_parallel(a : torch.Tensor, b : torch.Tensor, C : torch.Tensor, reg : float = 1.0, iters = 50):
    r"""
    The function of the Sinkhorn's algorithm with parallel processing, implemented with torch.

    Args:
        a (Tensor) - The batch of mass vectors of first distributions of shape (B,N,1), where B denotes the size of batch, N denotes the size of the first distributions.
        b (Tensor) - The batch of mass vectors of first distributions of shape (B,M,1), where M denotes the size of the second distributions.
        C (Tensor) - Cost matrix of shape (B,N,M) or (N,M). The former is for a batch of different cost matrices. the latter is for the case where a single cost matrix is used.
        reg (float) - The weight of regularization term (epsilon). (default: 1.0)
        iters (int) - The number of the iterations. (default: 50)

    Rtype:
        distances (Tensor) - The batch of the computed 1-Wasserstein distances. It is a vector of size B.
    """

    u = a
    v = b
    B = a.shape[0]
    N = a.shape[1]
    M = b.shape[1]
    P = torch.zeros(B, N, M).to(C.device)
    K = torch.exp(-(C / reg))

    for it in range(iters):
        u = a / (K @ v)
        if (C.dim() > 2):
            v = b / (K.transpose(1,2) @ u)
        else:
            v = b / (K.T @ u)

    du = torch.diag_embed(u.view(B,-1))
    du = du @ K
    du = du @ torch.diag_embed(v.view(B,-1))

    P = du#torch.diag_embed(u.view(B,-1)) @ K @ torch.diag_embed(v.view(B,-1))

    D = P * C

    return torch.sum(D.reshape(B,-1),dim = 1)

def pairwise_Minkowski_distance(X : torch.Tensor, p : int = 2):
        num_feats = X.shape[1]
        N = X.shape[0]
        def pairwise_norm(ids):
            X_a = X[ids[:, 0], :].reshape(-1, num_feats)
            X_b = X[ids[:, 1], :].reshape(-1, num_feats)
            distances = torch.norm(X_a - X_b, p=p, dim=1)
            return distances

        indices = torch.triu_indices(N, N)
        lists = pairwise_blocklize(indices.T, pairwise_norm, 100000)
        distances = torch.cat(lists, dim = 0)

        pairwise_D = torch.zeros(N, N).to(X.device)
        pairwise_D[indices[0, :], indices[1, :]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D
        return pairwise_D

class MLP(torch.nn.Module):
    r"""
    Custom multilayer perception network.

    Args:
        in_channels (int) - Number of dimensions of the input channel.
        out_channels (int) - Number of dimensions of the output channel.
        device - The computing device.
        hiddens (list[int]) - The list of numbers of dimensions of hidden layers. (default: [])
    """
    def __init__(self, in_channels : int, out_channels : int, device, hiddens : list = [], **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = len(hiddens)
        self.in_lin = torch.nn.Linear(in_channels, hiddens[0] if self.num_hiddens > 0 else out_channels).to(device)
        if self.num_hiddens > 0:
            modules = []
            for idx, dimension in enumerate(hiddens):
                modules.append(torch.nn.Linear(dimension, out_channels if idx + 1 >= self.num_hiddens else hiddens[idx + 1]).to(device))
            self.hidden_lin = torch.nn.ModuleList(modules)

    def forward(self, x):
        x = torch.relu(self.in_lin(x))
        if self.num_hiddens > 0:
            for l in self.hidden_lin:
                x = torch.relu(l(x))
        return x

class SubspaceOT(torch.nn.Module):
    r"""
    Deep dictionary learning framework for learning an OT-to-SOT mapping, which correspondings to the section 4.2 of our paper.

    Args:
        in_features (int) - Number of the dimensions of the input features.
        out_fearures (int) - Number of the dimensions of the output features, which is also the number of SOT basis vectors.
        device - The computing device.
        lasso_iters (int) - Number of the sparse coding layers, which is also the number of iterations of our modified solver based on the ISTA algorithm.

    Shapes:
        X (Tensor) - The feature matrix of shape (n,d), where n denotes the number of features, d denotes the number of dimensions of each feature.

    Rtypes:
        X_c (Tensor) - The sparse coding matrix of shape (n,k), where k denotes the number of basis vectors.
        X_p (Tensor) - The reconstructed features matrix of shape (n,d).
    """
    def __init__(self, in_features, out_features, device, lasso_iters = 10, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.lasso_maxIter = lasso_iters
        self.device = device

        self.basis = torch.nn.Parameter(torch.empty(in_features, out_features).to(device))
        self.soft_comp = torch.FloatTensor([0]).to(device)
        self.identity = torch.eye(out_features).to(device)
        self.lasso_c = torch.nn.Parameter(torch.FloatTensor([1]).to(device))
        self.lasso_lambda = torch.nn.Parameter(torch.FloatTensor([1]).to(device))
        self.mlp = MLP(in_features, 1, device, [math.ceil(0.5 * in_features)])#torch.nn.Linear(in_features, 1).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.basis, a = math.sqrt(5))
        self.lasso_c = torch.nn.Parameter(torch.norm(self.basis, p = 2) ** 2)

    def positive_soft_threshold(self, x, T):
        return torch.max(x - T, self.soft_comp)

    def sparse_coding(self, x, lamb, c, maxIter):
        # x: (N, D)
        # N is the number of features
        # D is the dimension of features

        N, M = x.shape[0], x.shape[1]
        pos_basis = torch.relu(self.basis)

        l = lamb / c
        DTy = pos_basis.T @ x.T # (K, N)
        S = self.identity - (1 / c) * (pos_basis.T @ pos_basis) # (K, K)
        z = torch.zeros(self.out_features, N).to(self.device) # (K, N)
        for t in range(maxIter):
            z = self.positive_soft_threshold(S @ z + (1 / c) * DTy, l)

        return z.T

    def get_basis(self):
        pos_basis = torch.relu(self.basis).T
        return pos_basis

    def forward(self, X):
        pos_basis = torch.relu(self.basis)

        lamb = self.mlp(X).reshape(-1)
        c = self.lasso_c
        X_c = self.sparse_coding(X, lamb, c, self.lasso_maxIter)

        return X_c, X_c @ pos_basis.T

    def subspace_cost_matrix(self, p = 2):
        def pairwise_norm(ids):
            V_a = V[ids[:, 0], :].reshape(-1, self.in_features)
            V_b = V[ids[:, 1], :].reshape(-1, self.in_features)
            distances = torch.norm(V_a - V_b, p=p, dim=1)
            return distances
        pos_basis = torch.relu(self.basis)
        V = pos_basis.T

        indices = torch.triu_indices(self.out_features, self.out_features)
        lists = pairwise_blocklize(indices.T, pairwise_norm, 100000)
        distances = torch.cat(lists, dim = 0)

        pairwise_D = torch.zeros(self.out_features, self.out_features).to(self.device)
        pairwise_D[indices[0, :], indices[1, :]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D

        # V = self.basis.T
        #ff = V.unsqueeze(1) - V.unsqueeze(0).abs().pow(p)
        #C = (V.unsqueeze(1) - V.unsqueeze(0)).abs().pow(p).sum(-1).pow(1/p)
        C = pairwise_D
        return C

    def loss_rc(self, X, X_p):
        N = X.shape[0]
        diff = X_p - X
        return torch.norm(diff, p = 2)**2 / N

    def loss_sp(self, X_c):
        N = X_c.shape[0]
        return ((torch.norm(X_c, p = 1, dim = 1) - 1)**2).sum() / N

    def loss_lp(self, X_c):
        N = X_c.shape[0]
        return ((torch.norm(X_c, p = 2, dim = 1) - 1)**2).sum() / N

class SubspaceOT_kmeans(SubspaceOT):
    r"""
    k-means framework for learning an OT-to-SOT mapping. It is a subclass from the SubspaceOT. However, in order to learn the mini-batch k-means, you should call its method partial_fit(X), where X denotes the inputted feature matrix.

    Args:
        batch_size (int) - The size of mini-batches for k-means learning.
        in_features (int) - Number of the dimensions of the input features.
        out_fearures (int) - Number of the dimensions of the output features, which is also the number of SOT basis vectors.
        device - The computing device.
        lasso_iters (int) - Number of the sparse coding layers, which is also the number of iterations of our modified solver based on the ISTA algorithm.

    Shapes:
        X (Tensor) - The feature matrix of shape (n,d), where n denotes the number of features, d denotes the number of dimensions of each feature.

    Rtypes:
        X_c (Tensor) - The sparse coding matrix of shape (n,k), where k denotes the number of basis vectors.
        X_p (Tensor) - The reconstructed features matrix of shape (n,d).
    """
    def __init__(self, batch_size, in_features, out_features, device, lasso_iters = 10, **kwargs):
        self.batch_size = batch_size
        self.batch_kmeans = MiniBatchKMeans(out_features, batch_size = batch_size)
        super().__init__(in_features, out_features, device, lasso_iters, **kwargs)
        return

    def partial_fit(self, X):
        def batch_fit(ids):
            a = X[ids[:,0],:].cpu().detach().numpy()
            self.batch_kmeans.partial_fit(a)

        indices = torch.arange(X.shape[0]).reshape(-1, 1)
        pairwise_blocklize(indices, batch_fit, self.batch_size)

    def subspace_cost_matrix(self, p = 2):
        def pairwise_norm(ids):
            V_a = V[ids[:, 0], :]
            V_b = V[ids[:, 1], :]
            distances = torch.norm(V_a - V_b, p=p, dim=1)
            return distances
        pos_basis = self.get_basis()
        V = pos_basis

        indices = torch.triu_indices(self.out_features, self.out_features)
        lists = pairwise_blocklize(indices.T, pairwise_norm, 100000)
        distances = torch.cat(lists, dim = 0)

        pairwise_D = torch.zeros(self.out_features, self.out_features).to(self.device)
        pairwise_D[indices[0, :], indices[1, :]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D

        C = pairwise_D
        return C

    def get_basis(self):
        pos_basis = torch.FloatTensor(self.batch_kmeans.cluster_centers_).to(self.device)
        return pos_basis

    def inertia(self):
        return self.batch_kmeans.inertia_

    def forward(self, X):
        pred = self.batch_kmeans.predict(X.cpu().detach().numpy())
        X_p = torch.FloatTensor(self.batch_kmeans.cluster_centers_[pred,:]).to(self.device)
        X_c = torch.LongTensor(pred).to(self.device)
        X_c = torch.nn.functional.one_hot(X_c, num_classes = self.out_features).float()

        return X_c, X_p

class SubspaceOT_ML(SubspaceOT):
    def __init__(self, in_features, out_features, ml_hiddens, device, lasso_iters = 10, **kwargs):
        super().__init__(in_features, out_features, device, lasso_iters, **kwargs)
        self.ml_hiddens = ml_hiddens
        self.ml_trans = torch.nn.Parameter(torch.empty(in_features, ml_hiddens).to(device))
        # self.lin1 = torch.nn.Linear(2*in_features, ml_hiddens).to(device)
        # self.lin2 = torch.nn.Linear(ml_hiddens, 1).to(device)
        self.reset_parameters_ml()
        return

    def reset_parameters_ml(self):
        torch.nn.init.kaiming_uniform_(self.ml_trans, a = math.sqrt(5))
        return

    def subspace_cost_matrix(self, p = 2):
        def pairwise_norm(ids):
            V_a = V[ids[:, 0], :].reshape(-1, self.ml_hiddens)
            V_b = V[ids[:, 1], :].reshape(-1, self.ml_hiddens)
            distances = torch.norm(V_a - V_b, p=p, dim=1)
            # V_a = V[ids[:, 0], :].reshape(-1, self.in_features)
            # V_b = V[ids[:, 1], :].reshape(-1, self.in_features)
            # V_mix = torch.sum(torch.relu(V_a - V_b), dim = 1)
            # V_mix = torch.cat((V_a,V_b), dim = 1)
            # V_mix = self.lin1(V_mix)
            # V_mix = self.lin2(V_mix)
            return distances
        pos_basis = torch.relu(self.basis)
        V = pos_basis.T
        V = V @ self.ml_trans

        indices = torch.triu_indices(self.out_features, self.out_features)
        lists = pairwise_blocklize(indices.T, pairwise_norm, 100000)
        distances = torch.cat(lists, dim = 0)

        pairwise_D = torch.zeros(self.out_features, self.out_features).to(self.device)
        pairwise_D[indices[0, :], indices[1, :]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D

        # V = self.basis.T
        #ff = V.unsqueeze(1) - V.unsqueeze(0).abs().pow(p)
        #C = (V.unsqueeze(1) - V.unsqueeze(0)).abs().pow(p).sum(-1).pow(1/p)
        C = pairwise_D
        return C

    def loss_ml(self, D, X_c):
        N = D.shape[0]
        D_prime = X_c @ self.subspace_cost_matrix() @ X_c.T
        return torch.norm(D - D_prime, p = 2) / (N * N)
    
class SubspaceOT_ML_ex(SubspaceOT):
    def __init__(self, in_features, out_features, ml_hiddens, device, lasso_iters = 10, epsilon = 0.01, **kwargs):
        super().__init__(in_features, out_features, device, lasso_iters, **kwargs)
        self.ml_hiddens = ml_hiddens
        self.ml_trans = torch.nn.Parameter(torch.empty(in_features, ml_hiddens).to(device))
        self.epsilon = epsilon
        self.lin1 = torch.nn.Linear(in_features, 2*in_features).to(device)
        self.lin2 = torch.nn.Linear(2*in_features, out_features).to(device)
        #self.C = torch.nn.Parameter(torch.empty(out_features, out_features).to(device))
        #self.mlp = MLP(in_features, out_features, device, [2*in_features])
        self.reset_parameters_ml()
        return

    def reset_parameters_ml(self):
        torch.nn.init.kaiming_uniform_(self.ml_trans, a = math.sqrt(5))
        #torch.nn.init.kaiming_uniform_(self.C, a = math.sqrt(5))
        return

    def subspace_cost_matrix(self, p = 2):
        def pairwise_norm(ids):
            V_a = V[ids[:, 0], :].reshape(-1, self.ml_hiddens)
            V_b = V[ids[:, 1], :].reshape(-1, self.ml_hiddens)
            distances = torch.norm(V_a - V_b, p=p, dim=1)
            # V_a = V[ids[:, 0], :].reshape(-1, self.in_features)
            # V_b = V[ids[:, 1], :].reshape(-1, self.in_features)
            # V_mix = torch.sum(torch.relu(V_a - V_b), dim = 1)
            # V_mix = torch.cat((V_a,V_b), dim = 1)
            # V_mix = self.lin1(V_mix)
            # V_mix = self.lin2(V_mix)
            return distances
        pos_basis = torch.relu(self.basis)
        V = pos_basis.T
        V = V @ self.ml_trans

        indices = torch.triu_indices(self.out_features, self.out_features)
        lists = pairwise_blocklize(indices.T, pairwise_norm, 100000)
        distances = torch.cat(lists, dim = 0)

        pairwise_D = torch.zeros(self.out_features, self.out_features).to(self.device)
        pairwise_D[indices[0, :], indices[1, :]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D

        # V = self.basis.T
        #ff = V.unsqueeze(1) - V.unsqueeze(0).abs().pow(p)
        #C = (V.unsqueeze(1) - V.unsqueeze(0)).abs().pow(p).sum(-1).pow(1/p)
        C = pairwise_D

        # C = torch.relu(self.C @ self.C.T)
        # return C - torch.diag_embed(torch.diagonal(C))
        return C
    
    def forward(self, X):
        pos_basis = torch.relu(self.basis)
        X_c = torch.relu(self.lin2(self.lin1(X))) + self.epsilon
        #X_c = torch.relu(self.mlp(X)) + self.epsilon
        X_c = torch.nn.functional.normalize(X_c, p = 1)

        return X_c, X_c @ pos_basis.T

    def loss_ml(self, D, X_c):
        N = D.shape[0]
        D_prime = X_c @ self.subspace_cost_matrix() @ X_c.T
        return torch.norm(D - D_prime, p = 2) / (N * N)

def asym_pairwise_subspaceOT_distance(model : SubspaceOT, X : torch.Tensor, batch_x : torch.LongTensor, Y : torch.Tensor, batch_y : torch.LongTensor, p : float = 2, reg : float = 1.0, iters : int = 50, return_loss : bool = False, block_size : int = 50000):
    r"""
    Function to compute the asymmetric pairwise 1-Wasserstein distances based on the learned SOT problems using the Sinkhorn's solver with parallelization.

    Args:
        model (SubspaceOT) - The learned SOT model.
        X (Tensor) - The first feature matrix of shape (n,d), where n denotes the number of features, d denotes the number of dimensions of each feature.
        batch_x (LongTensor) - The batch index vector of X of shape (n), which will be inputted into the torch_scatter.scatter_add function as the parameter of index.
        Y (Tensor) - The second feature matrix of shape (m,d).
        batch_y (LongTensor) - The batch index vector of Y of shape (m).
        p (float) - The order of norm for computing the Minkowski distance of each pair as the ground cost. (default: 2.0)
        reg (float) - The weight of regularization term (epsilon) of the Sinkhorn's algorithm. (default: 1.0)
        iters (int) - The number of the iterations of the Sinkhorn's algorithm. (default: 50)
        return_loss (bool) - If the function returns the losses.
        block_size (int) - The size of each block to blocklize the computation of distances.

    Rtypes:
        pairwise_distances (Tensor) - The distance matrix of shape (n,m).
        If return_loss is true:
            loss_lp (Tensor) - The l_p ball loss.
            loss_rc (Tensor) - The reconstruction loss.
            loss_sp (Tensor) - The simplex constraint violation loss.
    """
    def pairwise_sinkhorn(ids):
        mass_a = mass_x[ids[:,0],:].reshape(-1,D,1)
        mass_b = mass_y[ids[:,1],:].reshape(-1,D,1)
        distances = sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, iters)
        return distances
    X_c, X_p = model.forward(X)
    Y_c, Y_p = model.forward(Y)

    with torch.no_grad():
        mass_x = scatter_add(X_c, batch_x, dim = 0)
        mass_x = torch.nn.functional.normalize(mass_x, p=1)
        mass_y = scatter_add(Y_c, batch_y, dim = 0)
        mass_y = torch.nn.functional.normalize(mass_y, p=1)

        N = mass_x.shape[0]
        M = mass_y.shape[0]
        D = mass_x.shape[1]
        indices = torch.LongTensor([[ i for i in range(N) for j in range(M)], [j for i in range(N) for j in range(M)]])
        cost_mat = model.subspace_cost_matrix(p)

        # mass_a = mass_x[indices[0,:],:].reshape(-1,D,1)
        # mass_b = mass_y[indices[1,:],:].reshape(-1,D,1)

        # distances = self.sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, maxIter)
        lists = pairwise_blocklize(indices.T, pairwise_sinkhorn, block_size)
        distances = torch.cat(lists, dim=0)
        pairwise_D = torch.zeros(N,M).to(X.device)
        pairwise_D[indices[0,:], indices[1,:]] = distances

    if return_loss:
        loss_lp = (model.loss_lp(X_c) + model.loss_lp(Y_c)) * 0.5
        loss_rc = (model.loss_rc(X, X_p) + model.loss_rc(Y, Y_p)) * 0.5
        loss_sp = (model.loss_sp(X_c) + model.loss_sp(Y_c)) * 0.5
        return pairwise_D, loss_lp, loss_rc, loss_sp

    return pairwise_D

def pairwise_subspaceOT_distance(model : SubspaceOT, X : torch.Tensor, batch : torch.LongTensor, p : float = 2, reg : float = 1.0, iters : int = 50, return_loss : bool = False, return_time : bool = False, block_size : int = 50000):
    r"""
    Function to compute the symmetric pairwise 1-Wasserstein distances based on the learned SOT problems using the Sinkhorn's solver with parallelization.

    Args:
        model (SubspaceOT) - The learned SOT model.
        X (Tensor) - The feature matrix of shape (n,d), where n denotes the number of features, d denotes the number of dimensions of each feature.
        batch (LongTensor) - The batch index vector of X of shape (n), which will be inputted into the torch_scatter.scatter_add function as the parameter of index.
        p (float) - The order of norm for computing the Minkowski distance of each pair as the ground cost. (default: 2.0)
        reg (float) - The weight of regularization term (epsilon) of the Sinkhorn's algorithm. (default: 1.0)
        iters (int) - The number of the iterations of the Sinkhorn's algorithm. (default: 50)
        return_loss (bool) - If the function returns the losses.
        return_time (bool) - If the function returns the computational time costs.
        block_size (int) - The size of each block to blocklize the computation of distances.

    Rtypes:
        pairwise_distances (Tensor) - The distance matrix of shape (n,n).
        If return_loss is true:
            loss_lp (Tensor) - The l_p ball loss.
            loss_rc (Tensor) - The reconstruction loss.
            loss_sp (Tensor) - The simplex constraint violation loss.
        If return_time is true:
            time_list (list[3]) - The lists of the computational time costs of the encoding operation (time_list[0]), the distance computation operation (time_list[1]) and the loss computation operation (time_list[2]).
    """
    def pairwise_sinkhorn(ids):
        mass_a = mass[ids[:,0],:].reshape(-1,N,1)
        mass_b = mass[ids[:,1],:].reshape(-1,N,1)
        distances = sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, iters)
        return distances
    
    encoding_time = time.time()
    X_c, X_p = model.forward(X)
    encoding_time = time.time() - encoding_time

    distance_compute_time = time.time()
    with torch.no_grad():
        mass = scatter_add(X_c, batch, dim = 0)
        mass = torch.nn.functional.normalize(mass, p=1)
        B = mass.shape[0]
        N = mass.shape[1]
        indices = torch.triu_indices(B,B)
        cost_mat = model.subspace_cost_matrix(p)

        # mass_a = mass[indices[0,:],:].reshape(-1,N,1)
        # mass_b = mass[indices[1,:],:].reshape(-1,N,1)
        
        # distances = self.sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, maxIter)
        lists = pairwise_blocklize(indices.T, pairwise_sinkhorn, block_size)
        distances = torch.cat(lists, dim=0)
        pairwise_D = torch.zeros(B,B).to(X.device)
        pairwise_D[indices[0,:], indices[1,:]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D
    distance_compute_time = time.time() - distance_compute_time

    if return_loss:
        # if Laplacian is None:
        #     loss_li = self.loss_ml(X, X_c)#self.loss_li(X_c)
        # else:
        #     loss_li = self.loss_ml(X, X_c, Laplacian)
        loss_time = time.time()
        loss_lp = model.loss_lp(X_c)
        loss_rc = model.loss_rc(X, X_p)
        loss_sp = model.loss_sp(X_c)
        loss_time = time.time() - loss_time
        if return_time:
            return pairwise_D, loss_lp, loss_rc, loss_sp,[encoding_time, distance_compute_time, loss_time]
        else:
            return pairwise_D, loss_lp, loss_rc, loss_sp

    if return_time:
        return pairwise_D, [encoding_time, distance_compute_time, loss_time]
    else:
        return pairwise_D

def pairwise_wasserstein_distance_sinkhorn(X, batch, p = 2, reg = 1.0, maxIter = 50):
    r"""
    Function to compute the symmetric pairwise 1-Wasserstein distances of the original OT problems using the Sinkhorn's solver without parallelization.

    Args:
        X (Tensor) - The feature matrix of shape (n,d), where n denotes the number of features, d denotes the number of dimensions of each feature.
        batch (LongTensor) - The batch index vector of X of shape (n), which will be inputted into the torch_scatter.scatter_add function as the parameter of index.
        p (float) - The order of norm for computing the Minkowski distance of each pair as the ground cost. (default: 2.0)
        reg (float) - The weight of regularization term (epsilon) of the Sinkhorn's algorithm. (default: 1.0)
        iters (int) - The number of the iterations of the Sinkhorn's algorithm. (default: 50)

    Rtypes:
        pairwise_distances (Tensor) - The distance matrix of shape (n,n).
    """
    sp = split(batch)
    B = sp.shape[0]
    D = X.shape[1]
    indices = []

    start = 0
    for idx, each in enumerate(sp):
        end = start + each
        indices.append(start)
        start = end
    indices.append(start)

    pairwise_D = torch.zeros(B, B).to(X.device)

    for i in range(B):
        N = sp[i]
        a = torch.ones(N).to(X.device) / N
        X_1 = X[indices[i]:indices[i+1], :]

        for j in range(B-i):
            M = sp[i+j]
            b = torch.ones(M).to(X.device) / M
            X_2 = X[indices[i+j]:indices[i+j+1], :]

            if p == 0:
                C = torch.sign((X_1.unsqueeze(1) - X_2.unsqueeze(0)).abs().sum(-1))
            else:
                C = (X_1.unsqueeze(1) - X_2.unsqueeze(0)).abs().pow(p).sum(-1).pow(1/p)
            pairwise_D[i][i+j] = sinkhorn(a.reshape(-1, 1), b.reshape(-1, 1), C, reg, maxIter)

    diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
    pairwise_D = pairwise_D + pairwise_D.T - diag_D

    return pairwise_D