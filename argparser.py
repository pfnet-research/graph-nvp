import argparse
from distutils.util import strtobool


def get_parser():
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str, default='./data', help='Location for the dataset')
    parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'], help='dataset name')
    parser.add_argument('-f', '--data_file', type=str, default='qm9_relgcn_2.npz', help='Name of the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='data',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-t', '--save_interval', type=int, default=20,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=int, default=0,
                        help='Restore training from previous model checkpoint? 1 = Yes, 0 = No')
    parser.add_argument('--load_snapshot', type=str, default='', help='load the model from this path')
    # optimization
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
    parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
    parser.add_argument('-g', '--gpu', type=int, default=1, help='How many GPUs to distribute the training across?')
    parser.add_argument('--save_epochs', type=int, default=1, help='in how many epochs, a snapshot of the model'
                                                                   ' needs to be saved?')
    parser.add_argument('--communicator', type=str, default='hierarchical', help='The type of the communicator'
                                                                                 'to be used in chainermn')
    # evaluation
    parser.add_argument('--sample_batch_size', type=int, default=16,
                        help='How many samples to process in paralell during sampling?')
    # reproducibility
    parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
    parser.add_argument('--num_atoms', type=int, default=9, help='Maximum number of atoms in a molecule')
    parser.add_argument('--num_rels', type=int, default=4, help='Number of bond types')
    parser.add_argument('--num_atom_types', type=int, default=4, help='Types of atoms that can be used in a molecule')
    parser.add_argument('--num_node_masks', type=int, default=9,
                        help='Number of node masks to be used in coupling layers')
    parser.add_argument('--num_channel_masks', type=int, default=4,
                        help='Number of channel masks to be used in coupling layers')
    parser.add_argument('--num_node_coupling', type=int, default=12, help='Number of coupling layers with node masking')
    parser.add_argument('--num_channel_coupling', type=int, default=6,
                        help='Number of coupling layers with channel masking')
    parser.add_argument('--node_mask_size', type=int, default=5, help='Number of cells to be masked in the Node '
                                                                      'coupling layer')
    parser.add_argument('--channel_mask_size', type=int, default=-1, help='Number of cells to be masked in the Channel '
                                                                          'coupling layer')
    parser.add_argument('--apply_batch_norm', type=bool, default=False, help='Whether batch '
                                                                             'normalization should be performed')
    parser.add_argument('--debug', type=strtobool, default='false', help='To run training with more information')
    parser.add_argument('--learn_dist', type=strtobool, default='true', help='learn the distribution of feature matrix')
    parser.add_argument('--prior_var_adj', type=float, default=1.0,
                        help='Variance of the prior distribution for the adjacency matrix')
    parser.add_argument('--prior_var_x', type=float, default=1.0,
                        help='Variance of the prior distribution for the feature matrix')
    parser.add_argument('--additive_transformations', action='store_true', default=False,
                        help='apply only additive coupling layers')
    return parser
