from typing import Callable, Optional, Union, Tuple

import torch

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


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv_no_nn(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.model_name = 'GCNConv_no_nn'

        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

class GINConv_no_nn(MessagePassing):
    def __init__(self, num_layers, eps: float = 0., train_eps: bool = False,
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

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)


class ZeroMean(torch.nn.Module):
    def __init__(self, num_features, device, **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.sum = torch.zeros(num_features).to(device)
        self.mean = torch.zeros(num_features).to(device)
        self.n = 0
        self.num_features = num_features

    def forward(self, X, if_transform = False):
        if if_transform:
            return X - self.mean
        else:
            self.sum = self.sum + X.sum(dim = 0)
            self.n += X.shape[0]
            self.mean = self.sum / self.n
            return self.sum, self.n, self.mean

class PCAlayer(torch.nn.Module):
    def __init__(self, num_features, out_channel, device, **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.num_features = num_features
        self.out_channel = out_channel
        self.start_dim = num_features - out_channel
        self.end_dim = num_features
        self.zm = ZeroMean(num_features, device)
        self.XTX = torch.zeros(num_features, num_features).to(device)
        self.is_eig_computed = False
        self.eigvals = torch.zeros(num_features).to(device)
        self.eigvecs = torch.zeros(num_features, num_features).to(device)

    def forward(self, X, stage):
        if stage == 0:
            # stage 1: zero mean
            return self.zm(X, if_transform = False)
        elif stage == 1:
            # stage 2: compute X^T @ X
            X_m = self.zm(X, if_transform = True)
            self.XTX = self.XTX + X_m.T @ X_m
            return X_m
        elif stage == 2:
            # stage 3: compute eigvals & eigvec
            if not self.is_eig_computed:
                L, Q = torch.linalg.eigh(self.XTX)
                self.eigvals = L
                self.eigvecs = Q
                self.is_eig_computed = True

                # print("XTX:")
                # print(self.XTX.cpu().detach().numpy())
                # print("eigvals:")
                # print(self.eigvals.cpu().detach().numpy())
                # print("eigvecs:")
                # print(self.eigvecs.cpu().detach().numpy())
            return X
        elif stage == 3:
            # stage 4: PCA transform
            return X @ self.eigvecs[:,self.start_dim:self.end_dim]

class PCAlayer_batchmean(torch.nn.Module):
    def __init__(self, num_features, out_channel, device, **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.num_features = num_features
        self.out_channel = out_channel
        self.start_dim = num_features - out_channel
        self.end_dim = num_features
        self.XTX = torch.zeros(num_features, num_features).to(device)
        self.is_eig_computed = False
        self.eigvals = torch.zeros(num_features).to(device)
        self.eigvecs = torch.zeros(num_features, num_features).to(device)

    def forward(self, X, stage):
        if stage == 0:
            # stage 1: zero mean
            return
        elif stage == 1:
            # stage 2: compute X^T @ X
            X_m = X - torch.mean(X, dim = 0)
            self.XTX = self.XTX + X_m.T @ X_m
            return X_m
        elif stage == 2:
            # stage 3: compute eigvals & eigvec
            if not self.is_eig_computed:
                L, Q = torch.linalg.eigh(self.XTX)
                self.eigvals = L
                self.eigvecs = Q
                self.is_eig_computed = True

                # print("XTX:")
                # print(self.XTX.cpu().detach().numpy())
                # print("eigvals:")
                # print(self.eigvals.cpu().detach().numpy())
                # print("eigvecs:")
                # print(self.eigvecs.cpu().detach().numpy())
            return X
        elif stage == 3:
            # stage 4: PCA transform
            return X @ self.eigvecs[:,self.start_dim:self.end_dim]

class GraphPCA(torch.nn.Module):
    def __init__(self, MessageP: Callable, num_features, pca_hiddens, mlp_hiddens, device, batch_mean = True, **kwargs):
        super().__init__(**kwargs)
        self.mp = MessageP
        self.num_features = num_features
        self.num_layers = len(pca_hiddens)
        self.device = device
        self.each_layer_start_dim = []

        modules = []
        start = 0
        for idx, hidden in enumerate(pca_hiddens):
            self.each_layer_start_dim.append(start)
            if idx == 0:
                modules.append(PCAlayer_batchmean(self.num_features, hidden, device) if batch_mean else PCAlayer(self.num_features, hidden, device))
            else:
                modules.append(PCAlayer_batchmean(pca_hiddens[idx - 1], hidden, device) if batch_mean else PCAlayer(pca_hiddens[idx - 1], hidden, device))
            start += hidden
        self.each_layer_start_dim.append(start)
        self.pcas = torch.nn.ModuleList(modules)
        self.pcas_learned = [False for i in range(self.num_layers)]

        modules = []
        modules.append(torch.nn.Linear(self.each_layer_start_dim[-1], mlp_hiddens[0]).to(device))
        for idx, hidden in enumerate(mlp_hiddens):
            if idx + 1 < len(mlp_hiddens):
                modules.append(torch.nn.ReLU().to(device))
                modules.append(torch.nn.BatchNorm1d(hidden))
                modules.append(torch.nn.Linear(hidden, mlp_hiddens[idx + 1]))
        self.mlps = torch.nn.ModuleList(modules)

        self.cur_layer = 0
        self.cur_stage = -1
        self.cur_epoch = -1

    def forward(self, x, edge_index, epoch, graph_pooling = "no", batch = None, verbose = False):
        
        if epoch > self.cur_epoch and self.cur_layer < self.num_layers:
                
                self.cur_epoch = epoch
                if self.cur_stage >= 2:
                    self.pcas_learned[self.cur_layer] = True
                    self.cur_layer += 1
                    self.cur_stage = 0
                else:
                    self.cur_stage += 1

                if verbose and self.cur_layer < self.num_layers:
                    print(f"Epoch {epoch}: PCA Training, layer: {self.cur_layer + 1}, stage: {self.cur_stage + 1}")

        aggr_feature = torch.zeros(x.shape[0], self.each_layer_start_dim[-1]).to(self.device)

        if self.cur_stage < 2:

            for idx in range(self.num_layers):
                if self.pcas_learned[idx]:
                    x = self.mp(x,edge_index)
                    x = self.pcas[idx](x, stage = 3)
                    aggr_feature[:, self.each_layer_start_dim[idx] : self.each_layer_start_dim[idx + 1]] = x
                else:
                    x = self.mp(x,edge_index)
                    self.pcas[self.cur_layer](x, stage = self.cur_stage)
                    break
        elif self.cur_stage == 2:
            self.pcas[self.cur_layer](x, stage = 2)

        if self.cur_layer >= self.num_layers:
            # All layers are trained
            if not batch is None:
                if graph_pooling == "sum":
                    x = global_add_pool(aggr_feature, batch)
                elif graph_pooling == "max":
                    x = global_max_pool(aggr_feature, batch)
                elif graph_pooling == "mean":
                    x = global_mean_pool(aggr_feature, batch)

            for layers in self.mlps:
                x = layers(x)

        return x, self.cur_layer >= self.num_layers, False


class GraphPCA_translayer(torch.nn.Module):
    def __init__(self, MessageP: Callable, num_features, pca_hiddens, trans_epochs, trans_hiddens, mlp_hiddens, device, batch_mean = True, **kwargs):
        super().__init__(**kwargs)
        self.mp = MessageP
        self.num_features = num_features
        self.num_layers = len(pca_hiddens)
        self.trans_epochs = trans_epochs
        self.device = device
        self.each_layer_start_dim = []

        pca_list = []
        trans_list = []
        start = 0
        for i in range(self.num_layers):
            self.each_layer_start_dim.append(start)
            if i == 0:
                pca_list.append(PCAlayer_batchmean(self.num_features, pca_hiddens[0], device) if batch_mean else PCAlayer(self.num_features, device))
                trans_list.append(torch.nn.Parameter(torch.eye(pca_hiddens[0], trans_hiddens[0]).to(device)))
            else:
                pca_list.append(PCAlayer_batchmean(trans_hiddens[i-1], pca_hiddens[i], device) if batch_mean else PCAlayer(trans_hiddens[i-1], device))
                trans_list.append(torch.nn.Parameter(torch.eye(pca_hiddens[i], trans_hiddens[i]).to(device)))
            start += pca_hiddens[i]
        self.each_layer_start_dim.append(start)
        self.pcas = torch.nn.ModuleList(pca_list)
        self.pcas_learned = [False for i in range(self.num_layers)]
        self.trans = torch.nn.ParameterList(trans_list)
        self.trans_learned = [False for i in range(self.num_layers)]

        modules = []
        modules.append(torch.nn.Linear(self.each_layer_start_dim[-1], mlp_hiddens[0]).to(device))
        for idx, hidden in enumerate(mlp_hiddens):
            if idx + 1 < len(mlp_hiddens):
                modules.append(torch.nn.ReLU().to(device))
                modules.append(torch.nn.BatchNorm1d(hidden))
                modules.append(torch.nn.Linear(hidden, mlp_hiddens[idx + 1]))
        self.mlps = torch.nn.ModuleList(modules)

        self.cur_layer = 0
        self.cur_stage = -1
        self.cur_epoch = -1
        self.cur_trans_epoch = 1
        self.cur_trans_loss = 0
        self.cur_trans_n = 0

    def forward(self, x, edge_index, epoch, graph_pooling="no", batch=None, verbose=False):

        if epoch > self.cur_epoch and self.cur_layer < self.num_layers:

            self.cur_epoch = epoch
            if self.cur_stage >= 3:
                if self.cur_trans_epoch >= self.trans_epochs:
                    self.trans[self.cur_layer].requires_grad_(False)
                    self.pcas_learned[self.cur_layer] = True
                    self.cur_layer += 1
                    self.cur_stage = 0
                    self.cur_trans_epoch = 1
                else:
                    self.cur_trans_epoch = self.cur_trans_epoch + 1
            else:
                self.cur_stage += 1

            if verbose and self.cur_layer < self.num_layers:
                if self.cur_stage != 3:
                    print(f"Epoch {epoch}: PCA Training, layer: {self.cur_layer + 1}, stage: {self.cur_stage + 1}")
                elif self.cur_trans_n > 0:
                    print(f"Epoch {epoch}: translayer Training, layer: {self.cur_layer + 1}, stage: {self.cur_stage + 1}, loss: {self.cur_trans_loss/self.cur_trans_n}")
                    self.cur_trans_loss = 0
                    self.cur_trans_n = 0

        aggr_feature = torch.zeros(x.shape[0], self.each_layer_start_dim[-1]).to(self.device)

        if self.cur_stage != 2:

            for idx in range(self.num_layers):
                if self.pcas_learned[idx]:
                    x = self.mp(x, edge_index)
                    x = self.pcas[idx](x, stage=3)
                    aggr_feature[:, self.each_layer_start_dim[idx] : self.each_layer_start_dim[idx + 1]] = x
                    x = x @ self.trans[idx]
                    x = torch.nn.functional.normalize(x)
                else:
                    x = self.mp(x, edge_index)
                    self.pcas[self.cur_layer](x, stage=self.cur_stage)
                    break
        else:
            self.pcas[self.cur_layer](x, stage=2)

        if self.cur_stage == 3:
            x = self.mp(x, edge_index)
            x = self.pcas[self.cur_layer](x, stage=3)
            #t = x @ torch.eye(self.trans[self.cur_layer].shape[0], self.trans[self.cur_layer].shape[1]).to(self.device)
            x = x @ self.trans[self.cur_layer]
            #diff = torch.norm(x - t, p=1)
            self.cur_trans_n = self.cur_trans_n + x.shape[0] * x.shape[0]
            x = torch.nn.functional.normalize(x)
            x = x @ x.T - torch.eye(x.shape[0]).to(self.device)
            x = torch.norm(x, p = 1) #+ diff
            self.cur_trans_loss = self.cur_trans_loss + x

        if self.cur_layer >= self.num_layers:
            # All layers are trained
            if not batch is None:
                if graph_pooling == "sum":
                    x = global_add_pool(aggr_feature, batch)
                elif graph_pooling == "max":
                    x = global_max_pool(aggr_feature, batch)
                elif graph_pooling == "mean":
                    x = global_mean_pool(aggr_feature, batch)

            for layers in self.mlps:
                x = layers(x)

        return x, self.cur_layer >= self.num_layers, self.cur_stage == 3
        
        






