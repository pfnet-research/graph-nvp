import chainer
import chainer.functions as F
from chainer import cuda
from chainer_chemistry import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links import GraphLinear


def rescale_adj(adj):
    xp = cuda.get_array_module(adj)
    num_neighbors = F.sum(adj, axis=(1, 2))
    base = xp.ones(num_neighbors.shape, dtype=xp.float32)
    cond = num_neighbors.data != 0
    num_neighbors_inv = 1 / F.where(cond, num_neighbors, base)
    return adj * F.broadcast_to(num_neighbors_inv[:, None, None, :], adj.shape)


class RelGCNUpdate(chainer.Chain):

    def __init__(self, in_channels, out_channels, num_edge_type=4):
        """
        """
        super(RelGCNUpdate, self).__init__()
        with self.init_scope():
            self.graph_linear_self = GraphLinear(in_channels, out_channels)
            self.graph_linear_edge = GraphLinear(in_channels, out_channels * num_edge_type)
        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels

    def __call__(self, h, adj):
        """

        Args:
            h:
            adj:

        Returns:

        """

        mb, node, ch = h.shape

        # --- self connection, apply linear function ---
        hs = self.graph_linear_self(h)
        # --- relational feature, from neighbor connection ---
        # Expected number of neighbors of a vertex
        # Since you have to divide by it, if its 0, you need to arbitrarily set it to 1
        m = self.graph_linear_edge(h)
        m = F.reshape(m, (mb, node, self.out_ch, self.num_edge_type))
        m = F.transpose(m, (0, 3, 1, 2))
        # m: (batchsize, edge_type, node, ch)
        # hr: (batchsize, edge_type, node, ch)
        hr = F.matmul(adj, m)
        # hr: (batchsize, node, ch)
        hr = F.sum(hr, axis=1)
        return hs + hr


class RelGCNReadout(chainer.Chain):

    """RelGCN submodule for updates"""

    def __init__(self, in_channels, out_channels, nobias=True):
        super(RelGCNReadout, self).__init__()
        with self.init_scope():
            self.sig_linear = GraphLinear(
                in_channels, out_channels, nobias=nobias)
            self.tanh_linear = GraphLinear(
                in_channels, out_channels, nobias=nobias)

    def __call__(self, h, x=None):
        """Relational GCN

        (implicit: N = number of edges, R is number of types of relations)
        Args:
            h (chainer.Variable): (batchsize, num_nodes, ch)
                N x F : Matrix of edges, each row is a molecule and each column is a feature.
                F_l is the number of features at layer l
                F_0, the input layer, feature is type of molecule. Softmaxed

            x (chainer.Variable): (batchsize, num_nodes, ch)

        Returns:
            h_n (chainer.Variable): (batchsize, ch)
                F_n : Graph level representation

        Notes: I think they just incorporate "no edge" as one of the categories of relations, i've made it a separate
            tensor just to simplify some implementation, might change later
        """
        if x is None:
            in_feat = h
        else:
            in_feat = F.concat([h, x], axis=2)
        sig_feat = F.sigmoid(self.sig_linear(in_feat))
        tanh_feat = F.tanh(self.tanh_linear(in_feat))

        return F.tanh(F.sum(sig_feat * tanh_feat, axis=1))


class RelGCN(chainer.Chain):

    def __init__(self, out_channels=64, num_edge_type=4, ch_list=None,
                 n_atom_types=MAX_ATOMIC_NUM, input_type='int', scale_adj=False, activation=F.tanh):

        super(RelGCN, self).__init__()
        ch_list = ch_list or [16, 128, 64]
        # ch_list = [in_channels] + ch_list

        with self.init_scope():
            if input_type == 'int':
                self.embed = EmbedAtomID(out_size=ch_list[0], in_size=n_atom_types)
            elif input_type == 'float':
                self.embed = GraphLinear(None, ch_list[0])
            else:
                raise ValueError("[ERROR] Unexpected value input_type={}".format(input_type))
            self.rgcn_convs = chainer.ChainList(*[
                RelGCNUpdate(ch_list[i], ch_list[i+1], num_edge_type) for i in range(len(ch_list)-1)])
            self.rgcn_readout = RelGCNReadout(ch_list[-1], out_channels)
        # self.num_relations = num_edge_type
        self.input_type = input_type
        self.scale_adj = scale_adj
        self.activation = activation

    def __call__(self, x, adj):
        """

        Args:
            x: (batchsize, num_nodes, in_channels)
            adj: (batchsize, num_edge_type, num_nodes, num_nodes)

        Returns: (batchsize, out_channels)

        """
        if x.dtype == self.xp.int32:
            assert self.input_type == 'int'
        else:
            assert self.input_type == 'float'
        h = self.embed(x)  # (minibatch, max_num_atoms)
        if self.scale_adj:
            adj = rescale_adj(adj)
        for rgcn_conv in self.rgcn_convs:
            h = self.activation(rgcn_conv(h, adj))
        h = self.rgcn_readout(h)
        return h


if __name__ == '__main__':
    import numpy
    import numpy as np
    bs = 3
    nodes = 4
    ch = 5
    num_edge_type = 4
    x = np.zeros((bs, nodes, ch), dtype=np.float32)

    atom_size = 5
    out_dim = 4
    batch_size = 2
    # num_edge_type = 4
    out_ch = 128
    adj = np.zeros((bs, num_edge_type, nodes, nodes), dtype=np.float32)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)
    ).astype(numpy.int32)
    adj_data = numpy.random.randint(
        0, high=2, size=(batch_size, num_edge_type, atom_size, atom_size)
    ).astype(numpy.float32)

    rgcn = RelGCN(out_channels=out_ch)
    print('in', atom_data.shape, adj_data.shape)
    out = rgcn(atom_data, adj_data)
    print('out', out.shape)  # (bs, out_ch)
