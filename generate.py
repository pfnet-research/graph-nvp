import argparse
import os
from distutils.util import strtobool

import chainer
import numpy as np
from chainer.backends import cuda
from chainer.datasets import TransformDataset
from chainer_chemistry.datasets import NumpyTupleDataset
from rdkit.Chem import Draw, AllChem

from data import transform_qm9, transform_zinc250k
from data.transform_zinc250k import zinc250_atomic_num_list, transform_fn_zinc250k
from graph_nvp.hyperparams import Hyperparameters
from graph_nvp.utils import check_validity, adj_to_smiles, check_novelty, valid_mol, construct_mol
from utils.model_utils import load_model, get_latent_vec


def _to_numpy_array(a):
    if isinstance(a, chainer.Variable):
        a = a.array
    return cuda.to_cpu(a)


def generate_mols(model, temp=0.7, z_mu=None, batch_size=20, true_adj=None, gpu=-1):
    """

    :param model: GraphNVP model
    :param z_mu: latent vector of a molecule
    :param batch_size:
    :param true_adj:
    :param gpu:
    :return:
    """
    xp = np
    if gpu >= 0:
        xp = chainer.backends.cuda.cupy

    z_dim = model.adj_size + model.x_size
    mu = xp.zeros([z_dim], dtype=xp.float32)
    sigma_diag = xp.ones([z_dim])

    if model.hyperparams.learn_dist:
        sigma_diag = xp.sqrt(xp.exp(model.ln_var.data)) * sigma_diag
        # sigma_diag = xp.exp(xp.hstack((model.ln_var_x.data, model.ln_var_adj.data)))

    sigma = temp * sigma_diag

    with chainer.no_backprop_mode():
        if z_mu is not None:
            mu = z_mu
            sigma = 0.01 * xp.eye(z_dim, dtype=xp.float32)
        z = xp.random.normal(mu, sigma, (batch_size, z_dim)).astype(xp.float32)
        adj, x = model.reverse(z, true_adj=true_adj)
    return adj, x


def generate_mols_interpolation(model, z0=None, true_adj=None, gpu=-1, seed=0,
                                mols_per_row=13, delta=1.):
    np.random.seed(seed)
    latent_size = model.adj_size + model.x_size
    # TODO use learned variance of the model
    if z0 is None:
        mu = np.zeros([latent_size], dtype=np.float32)
        sigma = 0.02 * np.eye(latent_size, dtype=np.float32)
        z0 = np.random.multivariate_normal(mu, sigma).astype(np.float32)

    # z0 = np.random.normal(0., 0.1, (latent_size,)).astype(np.float32)

    # randomly generate 2 orthonormal axis x & y.
    x = np.random.randn(latent_size)
    x /= np.linalg.norm(x)

    y = np.random.randn(latent_size)
    y -= y.dot(x) * x
    y /= np.linalg.norm(y)

    num_mols_to_edge = mols_per_row // 2
    z_list = []
    for dx in range(-num_mols_to_edge, num_mols_to_edge + 1):
        for dy in range(-num_mols_to_edge, num_mols_to_edge + 1):
            z = z0 + x * delta * dx + y * delta * dy
            z_list.append(z)

    z_array = np.array(z_list, dtype=np.float32)
    if gpu >= 0:
        cuda.to_gpu(z_array, device=gpu)
    adj, x = model.reverse(z_array, true_adj=true_adj)
    return adj, x


def generate_mols_along_axis(model, z0=None, axis=None, n_mols=20, delta=0.1):
    z_list = []

    if z0 is None:
        temp = 0.7
        z_dim = model.adj_size + model.x_size
        mu = np.zeros([z_dim], dtype=np.float32)
        sigma_diag = np.ones([z_dim])

        if model.hyperparams.learn_dist:
            sigma_diag = np.sqrt(np.exp(model.ln_var.data)) * sigma_diag
        z0 = np.random.normal(mu, temp*sigma_diag, (z_dim)).astype(np.float32)

    for dx in range(n_mols):
        z = z0 + axis * delta * dx
        z_list.append(z)

    z_array = np.array(z_list, dtype=np.float32)

    with chainer.no_backprop_mode():
        adj, x = model.reverse(z_array)
    return adj, x


def visualize_interpolation(filepath, model, mol_smiles=None, mols_per_row=13,
                            delta=0.1, seed=0, atomic_num_list=[6, 7, 8, 9, 0], true_data=None, gpu=-1):
    z0 = None
    if mol_smiles is not None:
        z0 = get_latent_vec(model, mol_smiles)
    else:
        with chainer.no_backprop_mode():
            np.random.seed(seed)
            mol_index = np.random.randint(0, len(true_data))
            adj = np.expand_dims(true_data[mol_index][1], axis=0)
            x = np.expand_dims(true_data[mol_index][0], axis=0)
            z0 = model(adj, x)
            z0 = np.hstack((z0[0][0].data, z0[0][1].data)).squeeze(0)

    adj, x = generate_mols_interpolation(model, z0=z0, mols_per_row=mols_per_row, delta=delta, seed=seed, gpu=gpu)
    adj = _to_numpy_array(adj)
    x = _to_numpy_array(x)
    interpolation_mols = [valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list))
                          for x_elem, adj_elem in zip(x, adj)]
    valid_mols = [mol for mol in interpolation_mols if mol is not None]
    print('interpolation_mols valid {} / {}'
          .format(len(valid_mols), len(interpolation_mols)))
    img = Draw.MolsToGridImage(interpolation_mols, molsPerRow=mols_per_row, subImgSize=(250, 250))  # , useSVG=True
    img.save(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='./data/2-step-models')
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'], help='dataset name')
    parser.add_argument('--molecule_file', type=str, default='qm9_relgcn_kekulized_ggnp.npz',
                        help='path to molecule dataset')
    parser.add_argument("--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--hyperparams-path", type=str, default='graphnvp-hyperparams.json', required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument('--additive_transformations', type=strtobool, default='false',
                        help='apply only additive coupling layers')
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--n_experiments', type=int, default=1, help='number of times generation to be run')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature of the gaussian distribution')
    parser.add_argument('--draw_neighborhood', type=strtobool, default='true',
                        help='if neighborhood of a molecule to be visualized')
    parser.add_argument('--save_fig', type=strtobool, default='true')
    args = parser.parse_args()

    chainer.config.train = False
    snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
    print("loading hyperparamaters from {}".format(hyperparams_path))
    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params, debug=True)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    true_data = NumpyTupleDataset.load(os.path.join(args.data_dir, args.molecule_file))

    if args.data_name == 'qm9':
        atomic_num_list = [6, 7, 8, 9, 0]
        true_data = TransformDataset(true_data, transform_qm9.transform_fn)
        valid_idx = transform_qm9.get_val_ids()
    elif args.data_name == 'zinc250k':
        atomic_num_list = zinc250_atomic_num_list
        true_data = TransformDataset(true_data, transform_fn_zinc250k)
        valid_idx = transform_zinc250k.get_val_ids()

    train_idx = [t for t in range(len(true_data)) if t not in valid_idx]
    n_train = len(train_idx)
    train_idx.extend(valid_idx)
    train_data, _ = chainer.datasets.split_dataset(true_data, n_train, train_idx)
    train_adj = [a[1] for a in train_data]
    train_x = [a[0] for a in train_data]
    train_smiles = adj_to_smiles(train_adj, train_x, atomic_num_list)

    # 1. Random generation
    save_fig = args.save_fig
    valid_ratio = []
    unique_ratio = []
    novel_ratio = []
    for i in range(args.n_experiments):
        # 1. Random generation
        adj, x = generate_mols(model, batch_size=args.batch_size, true_adj=None, temp=args.temperature,
                               gpu=args.gpu)
        val_res = check_validity(adj, x, atomic_num_list, gpu=args.gpu)
        novel_ratio.append(check_novelty(val_res['valid_smiles'], train_smiles))
        unique_ratio.append(val_res['unique_ratio'])
        valid_ratio.append(val_res['valid_ratio'])
        n_valid = len(val_res['valid_mols'])

        # saves a png image of all generated molecules
        if save_fig:
            gen_dir = os.path.join(args.model_dir, 'generated')
            os.makedirs(gen_dir, exist_ok=True)
            filepath = os.path.join(gen_dir, 'generated_mols_{}.png'.format(i))
            img = Draw.MolsToGridImage(val_res['valid_mols'], legends=val_res['valid_smiles'],
                                       molsPerRow=20, subImgSize=(300, 300))  # , useSVG=True
            img.save(filepath)

    print("validity: mean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(valid_ratio), np.std(valid_ratio), valid_ratio))
    print("novelty: mean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(novel_ratio), np.std(novel_ratio), novel_ratio))
    print("uniqueness: mean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(unique_ratio), np.std(unique_ratio),
                                                                 unique_ratio))

    mol_smiles = None
    gen_dir = os.path.join(args.model_dir, 'generated')
    # 2. Intepolation generation
    if args.draw_neighborhood:
        for seed in [0, 1, 2, 3, 4]:
            filepath = os.path.join(gen_dir, 'generated_interpolation_molecules_seed{}.png'.format(seed))
            print('saving {}'.format(filepath))
            visualize_interpolation(filepath, model, mol_smiles=mol_smiles, mols_per_row=13, delta=args.delta,
                                    atomic_num_list=atomic_num_list, seed=seed, true_data=true_data, gpu=args.gpu)
