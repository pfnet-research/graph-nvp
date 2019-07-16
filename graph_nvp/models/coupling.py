import math

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L

from graph_nvp.models.mlp import MLP
from graph_nvp.models.relgcn import RelGCN


class Coupling(chainer.Chain):

    def __init__(self, num_nodes, num_relations, num_features, mask,
                 batch_norm=False):
        super(Coupling, self).__init__()
        self.mask = mask
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_features = num_features
        self.adj_size = self.num_nodes * self.num_nodes * self.num_relations
        self.x_size = self.num_nodes * self.num_features
        self.apply_batch_norm = batch_norm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reverse(self):
        raise NotImplementedError

    def to_gpu(self, device=None):
        super(Coupling, self).to_gpu(device)
        self.mask = cuda.to_gpu(self.mask, device)


class AffineAdjCoupling(Coupling):

    def __init__(self, num_nodes, num_relations, num_features, mask,
                 batch_norm=False, num_masked_cols=1,
                 ch_list=None):
        super(AffineAdjCoupling, self).__init__(num_nodes, num_relations, num_features, mask,
                                                batch_norm=batch_norm)
        self.num_masked_cols = num_masked_cols
        self.ch_list = ch_list
        self.adj_size = num_nodes * num_nodes * num_relations
        self.out_size = num_nodes * num_relations
        self.in_size = self.adj_size - self.out_size

        with self.init_scope():
            self.mlp = MLP(ch_list, in_size=self.in_size, activation=F.relu)
            self.lin = L.Linear(ch_list[-1], out_size=2 * self.out_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0., shape=[1])
            self.batch_norm = L.BatchNormalization(self.in_size)

    def __call__(self, adj):
        masked_adj = adj[:, :, self.mask]
        log_s, t = self._s_t_functions(masked_adj)
        t = F.broadcast_to(t, adj.shape)
        s = F.sigmoid(log_s + 2)
        s = F.broadcast_to(s, adj.shape)
        adj = adj * self.mask + adj * (s * ~self.mask) + t * (~self.mask)
        log_det_jacobian = F.sum(F.log(F.absolute(s)), axis=(1, 2, 3))
        return adj, log_det_jacobian

    def reverse(self, adj):
        masked_adj = adj[:, :, self.mask]
        log_s, t = self._s_t_functions(masked_adj)
        t = F.broadcast_to(t, adj.shape)
        s = F.sigmoid(log_s + 2)
        s = F.broadcast_to(s, adj.shape)
        adj = adj * self.mask + (((adj - t)/s) * ~self.mask)
        return adj, None

    def _s_t_functions(self, adj):
        x = F.reshape(adj, (adj.shape[0], -1))
        if self.apply_batch_norm:
            x = self.batch_norm(x)
        y = self.mlp(x)
        y = F.tanh(y)
        y = self.lin(y) * F.exp(self.scale_factor * 2)
        s = y[:, :self.out_size]
        t = y[:, self.out_size:]
        s = F.reshape(s, [y.shape[0], self.num_relations, self.num_nodes, 1])
        t = F.reshape(t, [y.shape[0], self.num_relations, self.num_nodes, 1])
        return s, t


class AdditiveAdjCoupling(Coupling):

    def __init__(self, num_nodes, num_relations, num_features, mask,
                 batch_norm=False,
                 num_masked_cols=1, ch_list=None):
        super(AdditiveAdjCoupling, self).__init__(num_nodes, num_relations,
                                                  num_features, mask,
                                                  batch_norm=batch_norm)
        self.num_masked_cols = num_masked_cols
        self.adj_size = num_nodes * num_nodes * num_relations
        self.out_size = num_nodes * num_relations
        self.in_size = self.adj_size - self.out_size

        with self.init_scope():
            self.mlp = MLP(ch_list, in_size=self.in_size, activation=F.relu)
            self.lin = L.Linear(ch_list[-1], out_size=self.out_size, initialW=1e-10)
            self.batch_norm = L.BatchNormalization(self.in_size)
            self.scale_factor = chainer.Parameter(initializer=0., shape=[1])

    def __call__(self, adj):
        """
        Performs one forward step of Adjacency coupling layer.
        Only an additive transformation is applied.
        :param adj: adjacency matrices of molecules
        :return: An adjacency matrix with additive transformation applied (with masking).
            Shape is same as the input adj.
        """
        masked_adj = adj[:, :, self.mask]
        t = self._s_t_functions(masked_adj)
        t = F.broadcast_to(t, adj.shape)
        adj += t * (~self.mask)
        return adj, chainer.Variable(self.xp.array(0.0, dtype="float32"))

    def reverse(self, adj):
        masked_adj = adj[:, :, self.mask]
        t = self._s_t_functions(masked_adj)
        # t = F.reshape(t, [t.shape[0], self.num_relations, self.num_nodes, 1])
        t = F.broadcast_to(t, adj.shape)
        adj -= t * (~self.mask)
        return adj, None

    def _s_t_functions(self, adj):
        adj = F.reshape(adj, (adj.shape[0], -1))
        x = adj
        if self.apply_batch_norm:
            x = self.batch_norm(x)
        y = self.mlp(x)
        y = F.tanh(y)
        y = self.lin(y) * F.exp(self.scale_factor * 2)

        y = F.reshape(y, [y.shape[0], self.num_relations, self.num_nodes, 1])
        return y


class AffineNodeFeatureCoupling(Coupling):

    def __init__(self, num_nodes, num_relations, num_features, mask,
                 batch_norm=False, input_type='float',
                 num_masked_cols=1, ch_list=None):
        super(AffineNodeFeatureCoupling, self).__init__(num_nodes, num_relations,
                                                        num_features, mask, batch_norm=batch_norm)
        self.num_masked_cols = num_masked_cols
        self.out_size = num_features * num_masked_cols
        with self.init_scope():
            self.rgcn = RelGCN(out_channels=ch_list['hidden'][0], num_edge_type=num_relations,
                               ch_list=ch_list['gcn'],
                               input_type=input_type, activation=F.relu)
            self.lin1 = L.Linear(ch_list['hidden'][0], out_size=ch_list['hidden'][1])
            self.lin2 = L.Linear(ch_list['hidden'][1], out_size=2*self.out_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0., shape=[1])
            self.batch_norm = L.BatchNormalization(ch_list['hidden'][0])

    def __call__(self, x, adj):
        masked_x = self.mask * x
        s, t = self._s_t_functions(masked_x, adj)
        x = masked_x + x * (s * ~self.mask) + t * ~self.mask
        log_det_jacobian = F.sum(F.log(F.absolute(s)), axis=(1, 2))
        return x, log_det_jacobian

    def reverse(self, y, adj):
        masked_y = self.mask * y
        s, t = self._s_t_functions(masked_y, adj)
        x = masked_y + (((y - t)/s) * ~self.mask)
        return x, None

    def _s_t_functions(self, x, adj):
        y = self.rgcn(x, adj)
        batch_size = x.shape[0]
        if self.apply_batch_norm:
            y = self.batch_norm(y)
        y = self.lin1(y)
        y = F.tanh(y)
        y = self.lin2(y) * F.exp(self.scale_factor*2)
        s = y[:, :self.out_size]
        t = y[:, self.out_size:]
        s = F.sigmoid(s + 2)

        t = F.reshape(t, [batch_size, 1, self.out_size])
        t = F.broadcast_to(t, [batch_size, int(self.num_nodes / self.num_masked_cols), self.out_size])
        s = F.reshape(s, [batch_size, 1, self.out_size])
        s = F.broadcast_to(s, [batch_size, int(self.num_nodes / self.num_masked_cols), self.out_size])
        return s, t


class AdditiveNodeFeatureCoupling(Coupling):
    def __init__(self, num_nodes, num_relations, num_features,
                 mask,
                 batch_norm=False, ch_list=None,
                 input_type='float', num_masked_cols=1):
        super(AdditiveNodeFeatureCoupling, self).__init__(num_nodes, num_relations,
                                                          num_features, mask,
                                                          batch_norm=batch_norm)
        self.num_masked_cols = num_masked_cols
        self.out_size = num_features * num_masked_cols

        with self.init_scope():
            self.rgcn = RelGCN(out_channels=ch_list['hidden'][0], num_edge_type=num_relations,
                               ch_list=ch_list['gcn'],
                               input_type=input_type, activation=F.relu)
            self.lin1 = L.Linear(ch_list['hidden'][0], out_size=ch_list['hidden'][1])
            self.lin2 = L.Linear(ch_list['hidden'][1], out_size=self.out_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0., shape=[1])
            self.batch_norm = L.BatchNormalization(ch_list['hidden'][0])

    def __call__(self, x, adj):
        masked_x = x * self.mask
        batch_size = x.shape[0]
        t = self._s_t_functions(masked_x, adj)
        t = F.reshape(t, [batch_size, 1, self.out_size])
        t = F.broadcast_to(t, [batch_size, int(self.num_nodes/self.num_masked_cols), self.out_size])
        if self.num_masked_cols > 1:
            t = F.reshape(t, [batch_size, self.num_nodes, self.num_features])
        x += t * ~self.mask
        return x, chainer.Variable(self.xp.array(0.0, dtype="float32"))

    def reverse(self, y, adj):
        masked_y = y * self.mask
        batch_size = y.shape[0]
        t = self._s_t_functions(masked_y, adj)
        t = F.reshape(t, [batch_size, 1, self.out_size])
        t = F.broadcast_to(t, [batch_size, int(self.num_nodes/self.num_masked_cols), self.out_size])
        if self.num_masked_cols > 1:
            t = F.reshape(t, [batch_size, self.num_nodes, self.num_features])
        y -= t * (~self.mask)
        return y, None

    def _s_t_functions(self, x, adj):
        y = self.rgcn(x, adj)
        if self.apply_batch_norm:
            y = self.batch_norm(y)
        y = self.lin1(y)
        y = F.tanh(y)
        y = self.lin2(y) * F.exp(self.scale_factor*2)
        return y
