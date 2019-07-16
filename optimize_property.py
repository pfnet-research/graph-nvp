import argparse
import os
from distutils.util import strtobool

import pickle

import chainer
import numpy as np
from chainer.datasets import TransformDataset
from chainer_chemistry.datasets import NumpyTupleDataset
from rdkit.Chem import Draw

from data import transform_qm9
from data.transform_zinc250k import zinc250_atomic_num_list, transform_fn_zinc250k
from generate import generate_mols_along_axis
from graph_nvp.hyperparams import Hyperparameters
from graph_nvp.utils import check_validity, construct_mol
from utils.model_utils import load_model, get_latent_vec
from utils.molecular_metrics import MolecularMetrics

from sklearn.linear_model import LinearRegression


def fit_model(model, atomic_num_list, data, property_name='qed'):
    batch_size = 1000
    max_samples = int(0.9 * len(data))
    true_vals = []
    z_data = []

    for i in range(0, max_samples, batch_size):
        print("processing batch: {}".format(i))
        true_batch = data[i:i+batch_size]
        true_mols = [construct_mol(a[0], a[1], atomic_num_list) for a in true_batch]
        true_mols = [m for m in true_mols if m]
        true_x = np.array([a[0] for a in true_batch])
        true_adj = np.array([a[1] for a in true_batch])
        z_batch = model(true_adj, true_x)
        z_batch = np.hstack((z_batch[0][0].data, z_batch[0][1].data))
        if property_name == 'qed':
            property_batch = MolecularMetrics.quantitative_estimation_druglikeness_scores(true_mols, norm=True)
        elif property_name == 'logp':
            property_batch = MolecularMetrics.water_octanol_partition_coefficient_scores(true_mols, norm=True)
        true_vals.append(property_batch)
        z_data.append(z_batch)

    z_data = np.vstack(z_data)
    true_vals = np.hstack(true_vals)
    linreg = LinearRegression(fit_intercept=True, normalize=True).fit(z_data, true_vals)
    print("R2 score:{:.2f}".format(linreg.score(z_data, true_vals)))
    return linreg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='./data/2-step-models', required=True)
    parser.add_argument("--data_dir", type=str, default='./data', required=True)
    parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'],
                        help='dataset name')
    parser.add_argument("--snapshot_path", "-snapshot", type=str, required=True)
    parser.add_argument("--hyperparams_path", type=str, default='graphnvp-hyperparams.json', required=True)
    parser.add_argument("--property_model_path", type=str, default=None)
    parser.add_argument('--molecule_file', type=str, default='qm9_relgcn_kekulized_ggnp.npz',
                        help='path to molecule dataset')
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--img_format", type=str, default='svg')
    parser.add_argument("--property_name", type=str, default='qed')
    parser.add_argument('--additive_transformations', type=strtobool, default=False,
                        help='apply only additive coupling layers')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature of the gaussian distributions')
    args = parser.parse_args()
    property_name = args.property_name.lower()
    chainer.config.train = False
    snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params, debug=True)

    true_data = NumpyTupleDataset.load(os.path.join(args.data_dir, args.molecule_file))

    if args.data_name == 'qm9':
        atomic_num_list = [6, 7, 8, 9, 0]
        true_data = TransformDataset(true_data, transform_qm9.transform_fn)
    elif args.data_name == 'zinc250k':
        atomic_num_list = zinc250_atomic_num_list
        true_data = TransformDataset(true_data, transform_fn_zinc250k)

    print("loading hyperparamaters from {}".format(hyperparams_path))
    property_model_path = os.path.join(args.model_dir, '{}_model.pkl'.format(property_name))
    if args.property_model_path:
        print("loading {} regression model from: {}".format(property_name, args.property_model_path))
        property_model = pickle.load(open(property_model_path, 'rb'))
    else:
        property_model = fit_model(model, atomic_num_list, true_data, property_name=property_name)
        print("saving {} regression model to: {}".format(property_name, property_model_path))
        pickle.dump(property_model, open(property_model_path, 'wb'))

    # mol_smiles = 'CCCc1ccccc1C=CC=CCNNC(=O)CCc1ccc(OC)c(C)c1'
    mol_smiles = 'CC1=C2C(=O)N(C)C12'
    axis = property_model.coef_/np.linalg.norm(property_model.coef_)
    z0 = get_latent_vec(model, mol_smiles, data_name=args.data_name)
    with chainer.no_backprop_mode():
        mol_index = 7969
        adj = np.expand_dims(true_data[mol_index][1], axis=0)
        x = np.expand_dims(true_data[mol_index][0], axis=0)
        z0 = model(adj, x)
        z0 = np.hstack((z0[0][0].data, z0[0][1].data)).squeeze(0)
    adj, x = generate_mols_along_axis(model, z0=z0, axis=axis, n_mols=100, delta=args.delta)
    interpolation_mols = check_validity(adj, x, atomic_num_list, return_unique=False)['valid_mols']

    if len(interpolation_mols) == 0:
        print("No valid molecules were generated")
        exit()

    if property_name == 'qed':
        property_vals = MolecularMetrics.quantitative_estimation_druglikeness_scores(interpolation_mols).tolist()
    elif property_name == 'logp':
        property_vals = MolecularMetrics.water_octanol_partition_coefficient_scores(interpolation_mols).tolist()

    property_legend = ['{} = {:.3f}'.format(property_name, prop) for prop in property_vals]

    gen_dir = os.path.join(args.model_dir, 'generated')
    os.makedirs(gen_dir, exist_ok=True)
    filepath = os.path.join(gen_dir, 'interpolated_mols_{}_delta_{}.{}'.format(property_name,
                                                                                args.delta,
                                                                               args.img_format))
    img = Draw.MolsToGridImage(interpolation_mols, molsPerRow=10, subImgSize=(250, 250),
                               legends=property_legend, useSVG=(args.img_format=='svg'))  # , useSVG=True
    if args.img_format == 'png':
        img.save(filepath)
    elif args.img_format == 'svg':
        with open(filepath, 'w') as img_file:
            img_file.write(img)