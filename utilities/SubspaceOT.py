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

from utilities.graphPCA import GINConv_no_nn

def split(batch):
    n = batch.shape[0]
    y = torch.ones((n,1))
    return scatter_add(y, batch.cpu(), dim = 0).cpu().detach().numpy().astype(int).reshape(-1)

def minmax_normalize(X):
    X_min = X.min()
    X_max = X.max()
    X = (X - X_min)/(X_max - X_min)
    return X

def Laplacian(d, lamb):
    return torch.exp(-lamb * d)

def pairwise_blocklize(indices, func, block_size, **kwargs):
    num_total = len(indices)
    num_blocks = int(math.floor(num_total / block_size))
    res = []
    for i in range(num_blocks + 1):
        if i*block_size >= num_total:
            break
        ret = func(indices[ i*block_size : num_total if (i+1)*block_size > num_total else (i+1)*block_size, :], **kwargs)
        res.append(ret)
    return res

def sinkhorn(a, b, C, reg = 1.0, maxIter = 50):
        u = a.reshape(-1,1)
        v = b.reshape(-1,1)
        N = a.shape[0]
        M = b.shape[0]
        P = torch.zeros(N,M).to(C.device)
        K = torch.exp(-(C / reg))

        for it in range(maxIter):
            u = a / (K @ v)
            v = b / (K.T @ u)

        P = torch.diag_embed(u.view(-1)) @ K @ torch.diag_embed(v.view(-1))

        return torch.sum(P * C)

def sinkhorn_parallel(a, b, C, reg = 1.0, maxIter = 50):
    # a - Mass vector of first distribution. Size: [B,N,1], B is the size of batch, N is the number of objects.
    # b - Mass vector of second distribution. Size: [B,M,1].
    # C - Cost matrix. Size: [B, N, M] or [N, M].

    u = a
    v = b
    B = a.shape[0]
    N = a.shape[1]
    M = b.shape[1]
    P = torch.zeros(B, N, M).to(C.device)
    K = torch.exp(-(C / reg))

    for it in range(maxIter):
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


class GNN_Null(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'GNN_Null'

    def forward(self, x, edge_index):
        return x

class GNN_WWL(MessagePassing):
    def __init__(self, num_layers, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.model_name = 'GNN_WWL'
        self.num_layers = num_layers

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""

        cat_out = [x]
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        deg = degree(edge_index[0], x[0].shape[0], dtype=torch.float).pow(-1.0)
        deg.masked_fill_(deg == float('inf'), 0)

        for i in range(self.num_layers):
            out = self.propagate(edge_index, x=x, size=size, deg = deg.reshape(-1, 1))

            x_r = x[1]
            if x_r is not None:
                out = 0.5 * out + 0.5 * x_r

            x: OptPairTensor = (out, out)

            cat_out.append(out)

        out = torch.cat(cat_out, dim = 1)

        return out

    def message(self, x_j: Tensor, deg_i: Tensor) -> Tensor:
        return x_j * deg_i

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

class GINConv_no_nn(MessagePassing):
    def __init__(self, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.model_name = 'GINConv_no_nn'
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        ## propagate_type: (x: OptPairTensor)
        # out = self.propagate(edge_index, x=x, size=size)

        # x_r = x[1]
        # if x_r is not None:
        #     out += (1 + self.eps) * x_r
        
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out = (1 + self.eps) * x_r


        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, device, hiddens = [], **kwargs):
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
    def __init__(self, in_features, out_features, device, Laplacian_lambda = 10, mlloss_norm = 1, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.Laplacian_lambda = Laplacian_lambda
        self.mlloss_norm = mlloss_norm
        self.device = device

        self.basis = torch.nn.Parameter(torch.empty(in_features, out_features).to(device))

        self.reset_parameters()

    def reset_parameters(self):
         torch.nn.init.kaiming_uniform_(self.basis, a = math.sqrt(5))

    def forward(self, X):
        pos_basis = torch.relu(self.basis)
        # Implementation with linalg.solve
        P_c = torch.linalg.solve(pos_basis.T @ pos_basis, pos_basis.T)
        # Implementation with inverse
        #P_c = (pos_basis.T @ pos_basis).inverse() @ pos_basis.T
        #P_c = (self.basis.T @ self.basis).inverse() @ self.basis.T
        X_c = torch.relu(X @ P_c.T)
        #P = self.basis @ P_c

        return X_c, X_c @ pos_basis.T

    def subspace_cost_matrix(self, p = 2):
        pos_basis = torch.relu(self.basis)
        V = pos_basis.T
        # V = self.basis.T
        #ff = V.unsqueeze(1) - V.unsqueeze(0).abs().pow(p)
        C = (V.unsqueeze(1) - V.unsqueeze(0)).abs().pow(p).sum(-1).pow(1/p)
        return C

    def sinkhorn(self, a, b, C, reg = 1.0, maxIter = 50):
        u = a.reshape(-1,1)
        v = b.reshape(-1,1)
        N = a.shape[0]
        M = b.shape[0]
        P = torch.zeros(N,M).to(self.device)
        K = torch.exp(-(C / reg))

        for it in range(maxIter):
            u = a / (K @ v)
            v = b / (K.T @ u)

        P = torch.diag_embed(u.view(-1)) @ K @ torch.diag_embed(v.view(-1))

        return torch.sum(P * C)

    def sinkhorn_parallel(self, a, b, C, reg = 1.0, maxIter = 50):
        # a - Mass vector of first distribution. Size: [B,N,1], B is the size of batch, N is the number of objects.
        # b - Mass vector of second distribution. Size: [B,M,1].
        # C - Cost matrix. Size: [B, N, M] or [N, M].

        u = a
        v = b
        B = a.shape[0]
        N = a.shape[1]
        M = b.shape[1]
        P = torch.zeros(B, N, M).to(self.device)
        K = torch.exp(-(C / reg))

        for it in range(maxIter):
            u = a / (K @ v)
            if (C.dim() > 2):
                v = b / (K.transpose(1,2) @ u)
            else:
                v = b / (K.T @ u)

        P = torch.diag_embed(u.view(B,-1)) @ K @ torch.diag_embed(v.view(B,-1))

        D = P * C
        return torch.sum(D.reshape(B,-1),dim = 1)

    def loss_li(self, X_c):
        N = X_c.shape[0]
        X_c = torch.nn.functional.normalize(X_c)
        covar = X_c @ X_c.T - torch.eye(N).to(self.device)
        return torch.norm(covar, p = 1) / (N * N)

    def Laplacian_pairwise_lp(self, X):
        def pairwise_norm(ids, X):
            X_a = X[ids[:, 0], :].reshape(-1, D)
            X_b = X[ids[:, 1], :].reshape(-1, D)
            distances = torch.norm(X_a - X_b, p=self.mlloss_norm, dim=1)
            return distances

        N = X.shape[0]
        D = X.shape[1]
        indices = torch.triu_indices(N, N)

        lists = pairwise_blocklize(indices.T, pairwise_norm, 2000000, X)
        distances = torch.cat(lists, dim = 0)
        # X_a = X[indices[0, :], :].reshape(-1, D)
        # X_b = X[indices[1, :], :].reshape(-1, D)
        # distances = torch.norm(X_a - X_b, p=self.mlloss_norm, dim=1)
        
        distances = Laplacian(distances, self.Laplacian_lambda)
        pairwise_D = torch.zeros(N, N).to(self.device)
        pairwise_D[indices[0, :], indices[1, :]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D
        return pairwise_D

    def loss_ml(self, X, X_c, Laplacian = None):
        N = X_c.shape[0]
        D = X.shape[1]
        X_c = torch.nn.functional.normalize(X_c)
        covar = X_c @ X_c.T

        if Laplacian is None:
            pairwise_D = self.Laplacian_pairwise_lp(X)
        else:
            pairwise_D = Laplacian

        return torch.norm(covar - pairwise_D, p = 1) / (N * N)

    def loss_rc(self, X, X_p):
        N = X.shape[0]
        diff = X_p - X
        return torch.norm(diff, p = 1) / N

    def loss_sp(self, X_c):
        N = X_c.shape[0]
        return torch.abs(torch.norm(X_c, p = 1) / N - 1)

    def asym_pairwise_subspaceOT_distance(self, X, batch_x, Y, batch_y, p = 2, reg = 1.0, maxIter = 50, return_loss = False):
        X_c, X_p = self.forward(X)
        Y_c, Y_p = self.forward(Y)

        mass_x = scatter_add(X_c, batch_x, dim = 0)
        mass_x = torch.nn.functional.normalize(mass_x)
        mass_y = scatter_add(Y_c, batch_y, dim = 0)
        mass_y = torch.nn.functional.normalize(mass_y)

        N = mass_x.shape[0]
        M = mass_y.shape[0]
        D = mass_x.shape[1]
        indices = torch.LongTensor([[ i for i in range(N) for j in range(M)], [j for i in range(N) for j in range(M)]])
        cost_mat = self.subspace_cost_matrix(p)

        mass_a = mass_x[indices[0,:],:].reshape(-1,D,1)
        mass_b = mass_y[indices[1,:],:].reshape(-1,D,1)

        distances = self.sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, maxIter)
        pairwise_D = torch.zeros(N,M).to(self.device)
        pairwise_D[indices[0,:], indices[1,:]] = distances

        if return_loss:
            loss_li = (self.loss_ml(X, X_c) + self.loss_ml(Y, Y_c)) * 0.5
            loss_rc = (self.loss_rc(X, X_p) + self.loss_rc(Y, Y_p)) * 0.5
            loss_sp = (self.loss_sp(X_c) + self.loss_sp(Y_c)) * 0.5
            return pairwise_D, loss_li, loss_rc, loss_sp

        return pairwise_D

    def pairwise_subspaceOT_distance(self, X, batch, p = 2, reg = 1.0, maxIter = 50, return_loss = False):
        X_c, X_p = self.forward(X)

        mass = scatter_add(X_c, batch, dim = 0)
        mass = torch.nn.functional.normalize(mass)
        B = mass.shape[0]
        N = mass.shape[1]
        indices = torch.triu_indices(B,B)
        cost_mat = self.subspace_cost_matrix(p)

        mass_a = mass[indices[0,:],:].reshape(-1,N,1)
        mass_b = mass[indices[1,:],:].reshape(-1,N,1)

        distances = self.sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, maxIter)
        pairwise_D = torch.zeros(B,B).to(self.device)
        pairwise_D[indices[0,:], indices[1,:]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D

        if return_loss:
            loss_li = self.loss_ml(X, X_c)#self.loss_li(X_c)
            loss_rc = self.loss_rc(X, X_p)
            loss_sp = self.loss_sp(X_c)
            return pairwise_D, loss_li, loss_rc, loss_sp

        return pairwise_D

    def pairwise_subspaceLp_distance(self, X, batch, p = 2, return_loss = False):
        X_c, X_p = self.forward(X)

        mass = scatter_add(X_c, batch, dim = 0)
        mass = torch.nn.functional.normalize(mass)
        B = mass.shape[0]
        N = mass.shape[1]
        indices = torch.triu_indices(B,B)

        mass_a = mass[indices[0,:],:].reshape(-1,N)
        mass_b = mass[indices[1,:],:].reshape(-1,N)

        distances = torch.norm(mass_a - mass_b, p = p, dim = 1) * 0.5
        pairwise_D = torch.zeros(B,B).to(self.device)
        pairwise_D[indices[0,:], indices[1,:]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D

        if return_loss:
            loss_li = self.loss_ml(X, X_c)#self.loss_li(X_c)
            loss_rc = self.loss_rc(X, X_p)
            loss_sp = self.loss_sp(X_c)
            return pairwise_D, loss_li, loss_rc, loss_sp

        return pairwise_D

    def pairwise_wasserstein_distance(self, X, batch, p = 2, reg = 1.0, maxIter = 50):
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

        pairwise_D = torch.zeros(B, B).to(self.device)

        for i in range(B):
            N = sp[i]
            a = torch.ones(N).to(self.device) / N
            X_1 = X[indices[i]:indices[i+1], :]

            for j in range(B-i):
                M = sp[i+j]
                b = torch.ones(M).to(self.device) / M
                X_2 = X[indices[i+j]:indices[i+j+1], :]

                if p == 0:
                    C = torch.sign((X_1.unsqueeze(1) - X_2.unsqueeze(0)).abs().sum(-1))
                else:
                    C = (X_1.unsqueeze(1) - X_2.unsqueeze(0)).abs().pow(p).sum(-1).pow(1/p)
                pairwise_D[i][i+j] = self.sinkhorn(a.reshape(-1, 1), b.reshape(-1, 1), C, reg, maxIter)

        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D

        return pairwise_D

    def pairwise_Lp_distance(self, X, batch, p = 2):

        mass = scatter_add(X, batch, dim = 0)
        mass = torch.nn.functional.normalize(mass)
        B = mass.shape[0]
        N = mass.shape[1]
        indices = torch.triu_indices(B,B)

        mass_a = mass[indices[0,:],:].reshape(-1,N)
        mass_b = mass[indices[1,:],:].reshape(-1,N)

        distances = torch.norm(mass_a - mass_b, p = p, dim = 1) * 0.5
        pairwise_D = torch.zeros(B,B).to(self.device)
        pairwise_D[indices[0,:], indices[1,:]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D

        return pairwise_D

class SubspaceOT_DKSVD(torch.nn.Module):
    def __init__(self, in_features, out_features, device, Laplacian_lambda = 10, mlloss_norm = 1, lasso_maxIter = 10, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.Laplacian_lambda = Laplacian_lambda
        self.mlloss_norm = mlloss_norm
        self.lasso_maxIter = lasso_maxIter
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

    # def sinkhorn(self, a, b, C, reg = 1.0, maxIter = 50):
    #     u = a.reshape(-1,1)
    #     v = b.reshape(-1,1)
    #     N = a.shape[0]
    #     M = b.shape[0]
    #     P = torch.zeros(N,M).to(self.device)
    #     K = torch.exp(-(C / reg))

    #     for it in range(maxIter):
    #         u = a / (K @ v)
    #         v = b / (K.T @ u)

    #     P = torch.diag_embed(u.view(-1)) @ K @ torch.diag_embed(v.view(-1))

    #     return torch.sum(P * C)

    # def sinkhorn_parallel(self, a, b, C, reg = 1.0, maxIter = 50):
    #     # a - Mass vector of first distribution. Size: [B,N,1], B is the size of batch, N is the number of objects.
    #     # b - Mass vector of second distribution. Size: [B,M,1].
    #     # C - Cost matrix. Size: [B, N, M] or [N, M].

    #     u = a
    #     v = b
    #     B = a.shape[0]
    #     N = a.shape[1]
    #     M = b.shape[1]
    #     P = torch.zeros(B, N, M).to(self.device)
    #     K = torch.exp(-(C / reg))

    #     for it in range(maxIter):
    #         u = a / (K @ v)
    #         if (C.dim() > 2):
    #             v = b / (K.transpose(1,2) @ u)
    #         else:
    #             v = b / (K.T @ u)

    #     du = torch.diag_embed(u.view(B,-1))
    #     du = du @ K
    #     du = du @ torch.diag_embed(v.view(B,-1))

    #     P = du#torch.diag_embed(u.view(B,-1)) @ K @ torch.diag_embed(v.view(B,-1))

    #     D = P * C

    #     return torch.sum(D.reshape(B,-1),dim = 1)

    def loss_li(self, X_c):
        N = X_c.shape[0]
        X_c = torch.nn.functional.normalize(X_c)
        covar = X_c @ X_c.T - torch.eye(N).to(self.device)
        return torch.norm(covar, p = 2)**2 / (N * N)

    def Laplacian_pairwise_lp(self, X):
        def pairwise_norm(ids):
            X_a = X[ids[:, 0], :].reshape(-1, D)
            X_b = X[ids[:, 1], :].reshape(-1, D)
            distances = torch.norm(X_a - X_b, p=self.mlloss_norm, dim=1)
            return distances
        N = X.shape[0]
        D = X.shape[1]
        indices = torch.triu_indices(N, N)

        # X_a = X[indices[0, :], :].reshape(-1, D)
        # X_b = X[indices[1, :], :].reshape(-1, D)
        # distances = torch.norm(X_a - X_b, p=self.mlloss_norm, dim=1)
        lists = pairwise_blocklize(indices.T, pairwise_norm, 50000)
        distances = torch.cat(lists, dim = 0)

        distances = Laplacian(distances, self.Laplacian_lambda)
        pairwise_D = torch.zeros(N, N).to(self.device)
        pairwise_D[indices[0, :], indices[1, :]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D
        return pairwise_D

    def loss_ml(self, X, X_c, Laplacian = None):
        N = X_c.shape[0]
        D = X.shape[1]
        X_c = torch.nn.functional.normalize(X_c)
        covar = X_c @ X_c.T

        if Laplacian is None:
            pairwise_D = self.Laplacian_pairwise_lp(X)
        else:
            pairwise_D = Laplacian

        return torch.norm(covar - pairwise_D, p = 2)**2 / (N * N)

    def loss_rc(self, X, X_p):
        N = X.shape[0]
        diff = X_p - X
        return torch.norm(diff, p = 2)**2 / N

    def loss_sp(self, X_c):
        N = X_c.shape[0]
        return ((torch.norm(X_c, p = 1, dim = 1) - 1)**2).sum() / N

    def loss_sp2(self, X_c):
        N = X_c.shape[0]
        return ((torch.norm(X_c, p = 2, dim = 1) - 1)**2).sum() / N

    def asym_pairwise_subspaceOT_distance(self, X, batch_x, Y, batch_y, p = 2, reg = 1.0, maxIter = 50, return_loss = False):
        def pairwise_sinkhorn(ids):
            mass_a = mass_x[ids[:,0],:].reshape(-1,D,1)
            mass_b = mass_y[ids[:,1],:].reshape(-1,D,1)
            distances = sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, maxIter)
            return distances
        X_c, X_p = self.forward(X)
        Y_c, Y_p = self.forward(Y)

        with torch.no_grad():
            mass_x = scatter_add(X_c, batch_x, dim = 0)
            mass_x = torch.nn.functional.normalize(mass_x, p=1)
            mass_y = scatter_add(Y_c, batch_y, dim = 0)
            mass_y = torch.nn.functional.normalize(mass_y, p=1)

            N = mass_x.shape[0]
            M = mass_y.shape[0]
            D = mass_x.shape[1]
            indices = torch.LongTensor([[ i for i in range(N) for j in range(M)], [j for i in range(N) for j in range(M)]])
            cost_mat = self.subspace_cost_matrix(p)

            # mass_a = mass_x[indices[0,:],:].reshape(-1,D,1)
            # mass_b = mass_y[indices[1,:],:].reshape(-1,D,1)

            # distances = self.sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, maxIter)
            lists = pairwise_blocklize(indices.T, pairwise_sinkhorn, 50000)
            distances = torch.cat(lists, dim=0)
            pairwise_D = torch.zeros(N,M).to(self.device)
            pairwise_D[indices[0,:], indices[1,:]] = distances

        if return_loss:
            #loss_li = (self.loss_ml(X, X_c) + self.loss_ml(Y, Y_c)) * 0.5
            loss_li = (self.loss_sp2(X_c) + self.loss_sp2(Y_c)) * 0.5
            loss_rc = (self.loss_rc(X, X_p) + self.loss_rc(Y, Y_p)) * 0.5
            loss_sp = (self.loss_sp(X_c) + self.loss_sp(Y_c)) * 0.5
            return pairwise_D, loss_li, loss_rc, loss_sp

        return pairwise_D

    def pairwise_subspaceOT_distance(self, X, batch, p = 2, reg = 1.0, maxIter = 50, return_loss = False, return_time = False, Laplacian = None):
        def pairwise_sinkhorn(ids):
            mass_a = mass[ids[:,0],:].reshape(-1,N,1)
            mass_b = mass[ids[:,1],:].reshape(-1,N,1)
            distances = sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, maxIter)
            return distances
        
        encoding_time = time.time()
        X_c, X_p = self.forward(X)
        encoding_time = time.time() - encoding_time

        distance_compute_time = time.time()
        with torch.no_grad():
            mass = scatter_add(X_c, batch, dim = 0)
            mass = torch.nn.functional.normalize(mass, p=1)
            B = mass.shape[0]
            N = mass.shape[1]
            indices = torch.triu_indices(B,B)
            cost_mat = self.subspace_cost_matrix(p)

            # mass_a = mass[indices[0,:],:].reshape(-1,N,1)
            # mass_b = mass[indices[1,:],:].reshape(-1,N,1)
            
            # distances = self.sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, maxIter)
            lists = pairwise_blocklize(indices.T, pairwise_sinkhorn, 10000)
            distances = torch.cat(lists, dim=0)
            pairwise_D = torch.zeros(B,B).to(self.device)
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
            loss_li = self.loss_sp2(X_c)
            loss_rc = self.loss_rc(X, X_p)
            loss_sp = self.loss_sp(X_c)
            loss_time = time.time() - loss_time
            if return_time:
                return pairwise_D, loss_li, loss_rc, loss_sp,[encoding_time, distance_compute_time, loss_time]
            else:
                return pairwise_D, loss_li, loss_rc, loss_sp

        if return_time:
            return pairwise_D, [encoding_time, distance_compute_time, loss_time]
        else:
            return pairwise_D

    def pairwise_subspaceLp_distance(self, X, batch, p = 2, return_loss = False):
        X_c, X_p = self.forward(X)

        with torch.no_grad():
            mass = scatter_add(X_c, batch, dim = 0)
            mass = torch.nn.functional.normalize(mass) * 0.5
            B = mass.shape[0]
            N = mass.shape[1]
            indices = torch.triu_indices(B,B)

            mass_a = mass[indices[0,:],:].reshape(-1,N)
            mass_b = mass[indices[1,:],:].reshape(-1,N)

            distances = torch.norm(mass_a - mass_b, p = p, dim = 1)
            pairwise_D = torch.zeros(B,B).to(self.device)
            pairwise_D[indices[0,:], indices[1,:]] = distances
            diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
            pairwise_D = pairwise_D + pairwise_D.T - diag_D

        if return_loss:
            loss_li = self.loss_ml(X, X_c)#self.loss_li(X_c)
            loss_rc = self.loss_rc(X, X_p)
            loss_sp = self.loss_sp(X_c)
            return pairwise_D, loss_li, loss_rc, loss_sp

        return pairwise_D

    def pairwise_wasserstein_distance(self, X, batch, p = 2, reg = 1.0, maxIter = 50):
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

        pairwise_D = torch.zeros(B, B).to(self.device)

        for i in range(B):
            N = sp[i]
            a = torch.ones(N).to(self.device) / N
            X_1 = X[indices[i]:indices[i+1], :]

            for j in range(B-i):
                M = sp[i+j]
                b = torch.ones(M).to(self.device) / M
                X_2 = X[indices[i+j]:indices[i+j+1], :]

                if p == 0:
                    C = torch.sign((X_1.unsqueeze(1) - X_2.unsqueeze(0)).abs().sum(-1))
                else:
                    C = (X_1.unsqueeze(1) - X_2.unsqueeze(0)).abs().pow(p).sum(-1).pow(1/p)
                pairwise_D[i][i+j] = sinkhorn(a.reshape(-1, 1), b.reshape(-1, 1), C, reg, maxIter)

        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D

        return pairwise_D

    def pairwise_Lp_distance(self, X, batch, p = 2):

        mass = scatter_add(X, batch, dim = 0)
        mass = torch.nn.functional.normalize(mass) * 0.5
        B = mass.shape[0]
        N = mass.shape[1]
        indices = torch.triu_indices(B,B)

        mass_a = mass[indices[0,:],:].reshape(-1,N)
        mass_b = mass[indices[1,:],:].reshape(-1,N)

        distances = torch.norm(mass_a - mass_b, p = p, dim = 1)
        pairwise_D = torch.zeros(B,B).to(self.device)
        pairwise_D[indices[0,:], indices[1,:]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D

        return pairwise_D

from sklearn.cluster import MiniBatchKMeans

class SubspaceOT_kmeans():
    def __init__(self, num_basis, batch_size):
        self.num_basis = num_basis
        self.batch_size = batch_size
        self.batch_kmeans = MiniBatchKMeans(num_basis, batch_size = batch_size)
        return

    def partial_fit(self, X):
        def batch_fit(ids):
            a = X[ids[:,0],:].cpu().detach().numpy()
            self.batch_kmeans.partial_fit(a)

        indices = torch.arange(X.shape[0]).reshape(-1, 1)
        pairwise_blocklize(indices, batch_fit, self.batch_size)

    def subspace_cost_matrix(self, device, p = 2):
        def pairwise_norm(ids):
            V_a = V[ids[:, 0], :]
            V_b = V[ids[:, 1], :]
            distances = torch.norm(V_a - V_b, p=p, dim=1)
            return distances
        pos_basis = torch.FloatTensor(self.batch_kmeans.cluster_centers_).to(device)
        V = pos_basis

        indices = torch.triu_indices(self.num_basis, self.num_basis)
        lists = pairwise_blocklize(indices.T, pairwise_norm, 50000)
        distances = torch.cat(lists, dim = 0)

        pairwise_D = torch.zeros(self.num_basis, self.num_basis).to(device)
        pairwise_D[indices[0, :], indices[1, :]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D

        # V = self.basis.T
        #ff = V.unsqueeze(1) - V.unsqueeze(0).abs().pow(p)
        #C = (V.unsqueeze(1) - V.unsqueeze(0)).abs().pow(p).sum(-1).pow(1/p)
        C = pairwise_D
        return C

    def get_basis(self, device):
        pos_basis = torch.FloatTensor(self.batch_kmeans.cluster_centers_).to(device)
        return pos_basis

    def transform(self, X):
        X_c = torch.LongTensor(self.batch_kmeans.predict(X.cpu().detach().numpy())).to(X.device)
        X_c = torch.nn.functional.one_hot(X_c, num_classes = self.num_basis).float()
        return X_c

    def asym_pairwise_subspaceOT_distance(self, X, batch_x, Y, batch_y, p = 2, reg = 1.0, maxIter = 50, return_loss = False):
        def pairwise_sinkhorn(ids):
            mass_a = mass_x[ids[:,0],:].reshape(-1,D,1)
            mass_b = mass_y[ids[:,1],:].reshape(-1,D,1)
            distances = sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, maxIter)
            return distances
        X_c = torch.LongTensor(self.batch_kmeans.predict(X.cpu().detach().numpy())).to(X.device)
        X_c = torch.nn.functional.one_hot(X_c, num_classes = self.num_basis).float()
        Y_c = torch.LongTensor(self.batch_kmeans.predict(Y.cpu().detach().numpy())).to(X.device)
        Y_c = torch.nn.functional.one_hot(Y_c, num_classes = self.num_basis).float()

        with torch.no_grad():
            mass_x = scatter_add(X_c, batch_x, dim = 0)
            mass_x = torch.nn.functional.normalize(mass_x, p=1)
            mass_y = scatter_add(Y_c, batch_y, dim = 0)
            mass_y = torch.nn.functional.normalize(mass_y, p=1)

            N = mass_x.shape[0]
            M = mass_y.shape[0]
            D = mass_x.shape[1]
            indices = torch.LongTensor([[ i for i in range(N) for j in range(M)], [j for i in range(N) for j in range(M)]])
            cost_mat = self.subspace_cost_matrix(X.device, p)

            # mass_a = mass_x[indices[0,:],:].reshape(-1,D,1)
            # mass_b = mass_y[indices[1,:],:].reshape(-1,D,1)

            # distances = self.sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, maxIter)
            lists = pairwise_blocklize(indices.T, pairwise_sinkhorn, 5000)
            distances = torch.cat(lists, dim=0)
            pairwise_D = torch.zeros(N,M).to(X.device)
            pairwise_D[indices[0,:], indices[1,:]] = distances

        return pairwise_D

    def pairwise_subspaceOT_distance(self, X, batch, p = 2, reg = 1.0, maxIter = 50, return_loss = False, Laplacian = None):
        def pairwise_sinkhorn(ids):
            mass_a = mass[ids[:,0],:].reshape(-1,N,1)
            mass_b = mass[ids[:,1],:].reshape(-1,N,1)
            distances = sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, maxIter)
            return distances

        X_c = torch.LongTensor(self.batch_kmeans.predict(X.cpu().detach().numpy())).to(X.device)
        X_c = torch.nn.functional.one_hot(X_c, num_classes = self.num_basis).float()

        with torch.no_grad():
            mass = scatter_add(X_c, batch, dim = 0)
            mass = torch.nn.functional.normalize(mass, p=1)
            B = mass.shape[0]
            N = mass.shape[1]
            indices = torch.triu_indices(B,B)
            cost_mat = self.subspace_cost_matrix(X.device, p)

            # mass_a = mass[indices[0,:],:].reshape(-1,N,1)
            # mass_b = mass[indices[1,:],:].reshape(-1,N,1)
            
            # distances = self.sinkhorn_parallel(mass_a, mass_b, cost_mat, reg, maxIter)
            lists = pairwise_blocklize(indices.T, pairwise_sinkhorn, 5000)
            distances = torch.cat(lists, dim=0)
            pairwise_D = torch.zeros(B,B).to(X.device)
            pairwise_D[indices[0,:], indices[1,:]] = distances
            diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
            pairwise_D = pairwise_D + pairwise_D.T - diag_D

        return pairwise_D   

class SOTGNN(torch.nn.Module):
    def __init__(self, in_features, SOT_hiddens, mlp_hiddens, device, Laplacian_lambda = 10, mlloss_norm = 1, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.Laplacian_lambda = Laplacian_lambda
        self.mlloss_norm = mlloss_norm
        self.device = device
        self.num_layers = len(SOT_hiddens)
        self.SOT_hiddens = SOT_hiddens
        self.mlp_hiddens = mlp_hiddens
        self.out_features = in_features

        self.gnn = GINConv_no_nn().to(device)

        modules = []
        for idx in range(len(SOT_hiddens)):
            self.out_features = self.out_features + SOT_hiddens[idx]
            if idx == 0:
                modules.append(SubspaceOT(in_features, SOT_hiddens[idx], device, Laplacian_lambda = Laplacian_lambda, mlloss_norm = mlloss_norm))
            else:
                modules.append(SubspaceOT(SOT_hiddens[idx-1], SOT_hiddens[idx], device, Laplacian_lambda = Laplacian_lambda, mlloss_norm = mlloss_norm))
        self.SOT = torch.nn.ModuleList(modules)

        modules = []
        modules.append(torch.nn.Linear(self.out_features, mlp_hiddens[0]).to(device))
        for idx, hidden in enumerate(mlp_hiddens):
            if idx + 1 < len(mlp_hiddens):
                modules.append(torch.nn.ReLU().to(device))
                modules.append(torch.nn.BatchNorm1d(hidden).to(device))
                modules.append(torch.nn.Linear(hidden, mlp_hiddens[idx + 1]).to(device))
        self.mlps = torch.nn.ModuleList(modules)

    def forward(self, x, edge_index, batch, graph_pooling = "no", Laplacian = None):

        loss_ml = 0
        loss_sp = 0
        loss_rc = 0

        aggr_feature = x

        for l in range(self.num_layers):
            x = self.gnn(x, edge_index)

            x_c, x_p = self.SOT[l](x)
            loss_ml = loss_ml + self.SOT[l].loss_ml(x, x_c, Laplacian)
            loss_sp = loss_sp + self.SOT[l].loss_sp(x_c)
            loss_rc = loss_rc + self.SOT[l].loss_rc(x, x_p)

            x = x_c

            aggr_feature = torch.cat((aggr_feature, x_c), dim = 1)

        loss_ml = loss_ml/self.num_layers
        loss_sp = loss_sp/self.num_layers
        loss_rc = loss_rc/self.num_layers

        if graph_pooling == "sum":
            x = global_add_pool(aggr_feature, batch)
        elif graph_pooling == "max":
            x = global_max_pool(aggr_feature, batch)
        elif graph_pooling == "mean":
            x = global_mean_pool(aggr_feature, batch)

        for l, layer in enumerate(self.mlps):
            x = layer(x)

        x = torch.nn.functional.softmax(x, dim = 1)

        return x, loss_ml, loss_sp, loss_rc

    def Laplacian_pairwise_lp(self, X):
        N = X.shape[0]
        D = X.shape[1]
        indices = torch.triu_indices(N, N)
        X_a = X[indices[0, :], :].reshape(-1, D)
        X_b = X[indices[1, :], :].reshape(-1, D)
        distances = torch.norm(X_a - X_b, p=self.mlloss_norm, dim=1)
        distances = Laplacian(distances, self.Laplacian_lambda)
        pairwise_D = torch.zeros(N, N).to(self.device)
        pairwise_D[indices[0, :], indices[1, :]] = distances
        diag_D = torch.diag_embed(torch.diagonal(pairwise_D))
        pairwise_D = pairwise_D + pairwise_D.T - diag_D
        return pairwise_D

class GINConv_no_nn_multi(torch.nn.Module):
    def __init__(self, num_layers, device, scale = 1.0, eps = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'GINConv_no_nn_multi'
        self.num_layers = num_layers
        self.gnn = GINConv_no_nn(eps).to(device)
        self.scale = scale
    
    def forward(self, x, edge_index):
        aggr_feature = x
        
        for l in range(self.num_layers):
            x = self.gnn(x, edge_index)
            aggr_feature = torch.cat((aggr_feature, x), dim = 1)

        return aggr_feature * self.scale
