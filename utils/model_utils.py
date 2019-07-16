import chainer
import numpy as np
from chainer_chemistry.dataset.preprocessors import GGNNPreprocessor
from rdkit import Chem

from data import transform_qm9
from data.transform_zinc250k import one_hot_zinc250k, transform_fn_zinc250k
from graph_nvp.models.model import GraphNvpModel


def load_model(snapshot_path, model_params, debug=False):
    print("loading snapshot: {}".format(snapshot_path))
    if debug:
        print("Hyper-parameters:")
        model_params.print()
    model = GraphNvpModel(model_params)
    if snapshot_path.endswith('.npz'):
        chainer.serializers.load_npz(snapshot_path, model)
    else:
        chainer.serializers.load_npz(
            snapshot_path, model, path='updater/optimizer:main/', strict=False)
    return model


def get_latent_vec(model, mol_smiles, data_name='qm9'):
    out_size = 9
    transform_fn = transform_qm9.transform_fn

    if data_name == 'zinc250k':
        out_size = 38
        transform_fn = transform_fn_zinc250k

    preprocessor = GGNNPreprocessor(out_size=out_size, kekulize=True)
    atoms, adj = preprocessor.get_input_features(Chem.MolFromSmiles(mol_smiles))
    atoms, adj, _ = transform_fn((atoms, adj, None))
    adj = np.expand_dims(adj, axis=0)
    atoms = np.expand_dims(atoms, axis=0)
    with chainer.no_backprop_mode():
        z = model(adj, atoms)
    z = np.hstack([z[0][0].data, z[0][1].data]).squeeze(0)
    return z
