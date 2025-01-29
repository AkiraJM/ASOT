import numpy as np
import torch
from torch_scatter import scatter_add
from .LowRankOT.LinSinkhorn import apply_lin_lr_lot
from .LowRankOT.utils import Square_Euclidean_Distance, factorized_square_Euclidean
import ot

def split(batch):
    n = batch.shape[0]
    y = torch.ones((n,1))
    return scatter_add(y, batch.cpu(), dim = 0).cpu().detach().numpy().astype(int).reshape(-1)

def LowRankOT(a, b, X, Y, r):
    Square_Euclidean_cost = lambda X, Y: Square_Euclidean_Distance(X, Y)
    Square_Euclidean_factorized_cost = lambda X, Y: factorized_square_Euclidean(X, Y)
    cost = Square_Euclidean_cost
    cost_factorized = Square_Euclidean_factorized_cost

    C = cost(X, Y)

    res, Q, R, g = apply_lin_lr_lot(
        X,
        Y,
        a,
        b,
        r,
        cost,
        cost_factorized,
        gamma_0=10,
        rescale_cost=True,
        time_out=50,
    )

    P_LOT = np.dot(Q / g, R.T)
    return np.sum(P_LOT * C)

def pairwise_wasserstein_distance(X, batch, p = 2, reg = 1.0, maxIter = 50, method = 'emd', n_projections = 1000, r = 100):
    sp = split(batch)
    B = sp.shape[0]
    D = X.shape[1]
    indices = []
    X_cpu = X.cpu().detach().numpy()

    start = 0
    for idx, each in enumerate(sp):
        end = start + each
        indices.append(start)
        start = end
    indices.append(start)

    pairwise_D = np.zeros((B, B))

    for i in range(B):
        if i == 0:
            continue

        N = sp[i]
        a = np.ones(N)/ N
        X_1 = X_cpu[indices[i]:indices[i+1], :]

        for j in range(B-i):
            if j == 0:
                continue
            M = sp[i+j]
            b = np.ones(M) / M
            X_2 = X_cpu[indices[i+j]:indices[i+j+1], :]

            if method != 'sliced':
                C = ot.dist(X_1, X_2, metric = 'minkowski', p = p)
            if method == 'emd':
                pairwise_D[i][i+j] = ot.emd2(a, b, C)
            elif method == 'sinkhorn':
                pairwise_D[i][i+j] = ot.sinkhorn2(a, b, C, reg, numItermax= maxIter)
            elif method == 'sliced':
                pairwise_D[i][i+j] = ot.sliced.sliced_wasserstein_distance(X_1, X_2, a, b, p = 1, n_projections =n_projections)
            elif method == 'lr':
                pairwise_D[i][i+j] = LowRankOT(a,b,X_1,X_2,r)
            else:
                #pairwise_D[i][i+j] = ot.gaussian.empirical_bures_wasserstein_distance(X_1, X_2)
                pairwise_D[i][i+j] = np.linalg.norm(np.mean(X_1, axis = 0) - np.mean(X_2, axis = 0), ord = p)

    pairwise_D = pairwise_D + pairwise_D.T

    return pairwise_D

def pairwise_SOT_distance(basis, mass_vecotrs, cost_matrix = None, p = 2, reg = 1.0, maxIter = 50, method = 'emd', n_projections = 1000, r = 100):
    B = mass_vecotrs.shape[0]
    D = basis.shape[1]
    K = basis.shape[0]
    indices = []
    basis_cpu = basis.cpu().detach().numpy().astype(np.float64)
    mass_cpu = mass_vecotrs.cpu().detach().numpy().astype(np.float64)
    mass_norm = np.linalg.norm(mass_cpu, ord = 1, axis = 1)
    mass_cpu = mass_cpu/mass_norm.reshape(-1,1)

    if cost_matrix is None:
        C_basis = ot.dist(basis_cpu, basis_cpu, metric = 'minkowski', p = p)
    else:
        C_basis = cost_matrix

    pairwise_D = np.zeros((B, B))

    for i in range(B):
        if i == 0:
            continue

        a = mass_cpu[i,:]
        X_1 = basis_cpu

        for j in range(B-i):
            if j == 0:
                continue

            b = mass_cpu[i+j,:]
            X_2 = basis_cpu

            if method == 'emd':
                pairwise_D[i][i+j] = ot.emd2(a, b, C_basis)
            elif method == 'sinkhorn':
                pairwise_D[i][i+j] = ot.sinkhorn2(a, b, C_basis, reg, numItermax= maxIter)
            elif method == 'sliced':
                pairwise_D[i][i+j] = ot.sliced.sliced_wasserstein_distance(X_1, X_2, a, b, p = 1, n_projections =n_projections)
            elif method == 'lr':
                pairwise_D[i][i+j] = LowRankOT(a,b,X_1,X_2,r)
            else:
                pairwise_D[i][i+j] = np.linalg.norm(np.mean(X_1, axis = 0) - np.mean(X_2, axis = 0), ord = p)

    pairwise_D = pairwise_D + pairwise_D.T

    return pairwise_D

def asym_pairwise_wasserstein_distance(X, batch, X2, batch2, p = 2, reg = 1.0, maxIter = 50, method = 'emd', n_projections = 1000, r = 100):
    sp = split(batch)
    B = sp.shape[0]
    D = X.shape[1]
    indices = []
    X_cpu = X.cpu().detach().numpy()

    start = 0
    for idx, each in enumerate(sp):
        end = start + each
        indices.append(start)
        start = end
    indices.append(start)

    sp2 = split(batch2)
    B2 = sp2.shape[0]
    indices2 = []
    X_cpu2 = X2.cpu().detach().numpy()

    start = 0
    for idx, each in enumerate(sp2):
        end = start + each
        indices2.append(start)
        start = end
    indices2.append(start)
    pairwise_D = np.zeros((B, B2))

    for i in range(B):
        N = sp[i]
        a = np.ones(N)/ N
        X_1 = X_cpu[indices[i]:indices[i+1], :]

        for j in range(B2):
            M = sp2[j]
            b = np.ones(M) / M
            X_2 = X_cpu2[indices2[j]:indices2[j+1], :]

            if method != 'sliced':
                C = ot.dist(X_1, X_2, metric = 'minkowski', p = p)
            if method == 'emd':
                pairwise_D[i][j] = ot.emd2(a, b, C)
            elif method == 'sinkhorn':
                pairwise_D[i][j] = ot.sinkhorn2(a, b, C, reg, numItermax= maxIter)
            elif method == 'sliced':
                pairwise_D[i][j] = ot.sliced.sliced_wasserstein_distance(X_1, X_2, a, b, p = 1, n_projections =n_projections)
            elif method == 'lr':
                pairwise_D[i][j] = LowRankOT(a,b,X_1,X_2,r)
            else:
                pairwise_D[i][j] = np.linalg.norm(np.mean(X_1, axis = 0) - np.mean(X_2, axis = 0), ord = p)

    return pairwise_D  