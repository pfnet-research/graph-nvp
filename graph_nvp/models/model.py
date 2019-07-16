import chainer
import chainer.links as L
import chainer.functions as F

from graph_nvp.hyperparams import Hyperparameters
from graph_nvp.models.coupling import AffineNodeFeatureCoupling, AffineAdjCoupling, AdditiveNodeFeatureCoupling, \
    AdditiveAdjCoupling


class GraphNvpModel(chainer.Chain):

    def __init__(self, hyperparams: Hyperparameters):
        super(GraphNvpModel, self).__init__()
        self.hyperparams = hyperparams
        self._init_params(hyperparams)
        self._need_initialization = False
        if self.masks is None:
            self._need_initialization = True
            self.masks = dict()
            self.masks['node'] = self._create_masks('node')
            self.masks['channel'] = self._create_masks('channel')
            self.hyperparams.masks = self.masks
        self.adj_size = self.num_nodes * self.num_nodes * self.num_relations
        self.x_size = self.num_nodes * self.num_features

        with self.init_scope():
            if hyperparams.learn_dist:
                self.ln_var = chainer.Parameter(initializer=0., shape=[1])
            else:
                self.ln_var = chainer.Variable(initializer=0., shape=[1])

            # AffineNodeFeatureCoupling found to be unstable.
            channel_coupling = AdditiveNodeFeatureCoupling
            node_coupling = AffineAdjCoupling
            if self.hyperparams.additive_transformations:
                channel_coupling = AdditiveNodeFeatureCoupling
                node_coupling = AdditiveAdjCoupling
            clinks = [
                channel_coupling(self.num_nodes, self.num_relations, self.num_features,
                                 self.masks['channel'][i % hyperparams.num_masks['channel']],
                                 num_masked_cols=int(self.num_nodes / self.hyperparams.num_masks['channel']),
                                 ch_list=hyperparams.gnn_channels,
                                 batch_norm=hyperparams.apply_batch_norm)
                for i in range(hyperparams.num_coupling['channel'])]

            clinks.extend([
                node_coupling(self.num_nodes, self.num_relations, self.num_features,
                              self.masks['node'][i % hyperparams.num_masks['node']],
                              num_masked_cols=int(self.num_nodes / self.hyperparams.num_masks['channel']),
                              batch_norm=hyperparams.apply_batch_norm,
                              ch_list=hyperparams.mlp_channels)
                for i in range(hyperparams.num_coupling['node'])])
            self.clinks = chainer.ChainList(*clinks)

    def __call__(self, adj, x):
        h = F.broadcast(x)
        # add uniform noise to node feature matrices
        if chainer.config.train:
            h += self.xp.random.uniform(0, 0.9, x.shape)
        adj = F.broadcast(adj)
        sum_log_det_jacs_x = F.broadcast(self.xp.zeros([h.shape[0]], dtype=self.xp.float32))
        sum_log_det_jacs_adj = F.broadcast(self.xp.zeros([h.shape[0]], dtype=self.xp.float32))
        # forward step of channel-coupling layers
        for i in range(self.hyperparams.num_coupling['channel']):
            h, log_det_jacobians = self.clinks[i](h, adj)
            sum_log_det_jacs_x += log_det_jacobians
        # add uniform noise to adjacency tensors
        if chainer.config.train:
            adj += self.xp.random.uniform(0, 0.9, adj.shape)
        # forward step of adjacency-coupling
        for i in range(self.hyperparams.num_coupling['channel'], len(self.clinks)):
            adj, log_det_jacobians = self.clinks[i](adj)
            sum_log_det_jacs_adj += log_det_jacobians

        adj = F.reshape(adj, (adj.shape[0], -1))
        h = F.reshape(h, (h.shape[0], -1))
        out = [h, adj]
        return out, [sum_log_det_jacs_x, sum_log_det_jacs_adj]

    def reverse(self, z, true_adj=None):
        """
        Returns a molecule, given its latent vector.
        :param z: latent vector. Shape: [B, N*N*M + N*T]
            B = Batch size, N = number of atoms, M = number of bond types,
            T = number of atom types (Carbon, Oxygen etc.)
        :param true_adj: used for testing. An adjacency matrix of a real molecule
        :return: adjacency matrix and feature matrix of a molecule
        """
        batch_size = z.shape[0]

        with chainer.no_backprop_mode():
            z_x = chainer.as_variable(z[:, :self.x_size])
            z_adj = chainer.as_variable(z[:, self.x_size:])

            if true_adj is None:
                h_adj = F.reshape(z_adj, (batch_size, self.num_relations, self.num_nodes, self.num_nodes))

                # First, the adjacency coupling layers are applied in reverse order to get h_adj
                for i in reversed(range(self.hyperparams.num_coupling['channel'], len(self.clinks))):
                    h_adj, log_det_jacobians = self.clinks[i].reverse(h_adj)

                # make adjacency matrix from h_adj
                adj = h_adj
                adj += self.xp.transpose(adj, (0, 1, 3, 2))
                adj = adj / 2
                adj = F.softmax(adj, axis=1)
                max_bond = F.repeat(F.max(adj, axis=1).reshape(batch_size, -1, self.num_nodes, self.num_nodes),
                                    self.num_relations, axis=1)
                adj = F.floor(adj / max_bond)
            else:
                adj = true_adj

            h_x = F.reshape(z_x, (batch_size, self.num_nodes, self.num_features))

            # channel coupling layers
            for i in reversed(range(self.hyperparams.num_coupling['channel'])):
                h_x, log_det_jacobians = self.clinks[i].reverse(h_x, adj)

        return adj, h_x

    def _init_params(self, hyperparams):
        self.num_nodes = hyperparams.num_nodes
        self.num_relations = hyperparams.num_relations
        self.num_features = hyperparams.num_features
        self.masks = hyperparams.masks

    def _create_masks(self, type):
        masks = []
        num_cols = int(self.num_nodes / self.hyperparams.num_masks[type])
        if type == 'node':
            # Columns of the adjacency matrix is masked
            for i in range(self.hyperparams.num_masks[type]):
                node_mask = self.xp.ones([self.num_nodes, self.num_nodes], dtype=self.xp.bool)
                for j in range(num_cols):
                    node_mask[:, i + j] = False
                masks.append(node_mask)
        elif type == 'channel':
            # One row (one node) of the feature matrix is masked
            num_cols = int(self.num_nodes / self.hyperparams.num_masks[type])
            for i in range(self.hyperparams.num_masks[type]):
                ch_mask = self.xp.ones([self.num_nodes, self.num_features], dtype=self.xp.bool)
                for j in range(num_cols):
                    ch_mask[i * num_cols + j, :] = False
                masks.append(ch_mask)
        return masks

    def log_prob(self, z, logdet):
        logdet[0] = logdet[0] - self.x_size
        logdet[1] = logdet[1] - self.adj_size
        ln_var_adj = self.ln_var * self.xp.ones([self.adj_size])
        ln_var_x = self.ln_var * self.xp.ones([self.x_size])
        nll_adj = F.average(F.sum(F.gaussian_nll(z[1], self.xp.zeros([self.adj_size], dtype=self.xp.float32),
                                                 ln_var_adj, reduce='no'), axis=1) - logdet[1])
        nll_adj /= self.adj_size

        nll_x = F.average(F.sum(F.gaussian_nll(z[0], self.xp.zeros([self.x_size], dtype=self.xp.float32),
                                               ln_var_x, reduce='no'), axis=1) - logdet[0])
        nll_x /= self.x_size
        if nll_x.array < 0:
            print('nll_x:{}'.format(nll_x))

        return [nll_x, nll_adj]

    def save_hyperparams(self, path):
        self.hyperparams.save(path)

    def load_hyperparams(self, path):
        """
        loads hyper parameters from a json file
        :param path:
        :return:
        """
        hyperparams = Hyperparameters(path=path)
        self._init_params(hyperparams)

    def to_gpu(self, device=None):
        super(GraphNvpModel, self).to_gpu(device)

