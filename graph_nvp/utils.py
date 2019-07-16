import chainer
import chainer.functions as F
import numpy as np
from chainer.backends import cuda
from rdkit import Chem
from rdkit.Chem import Draw

atom_decoder_m = {0: 6, 1: 7, 2: 8, 3: 9}
bond_decoder_m = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}


def flatten_graph_data(adj, x):
    return F.hstack((F.reshape(adj,[adj.shape[0], -1]),
                     F.reshape(x, [x.shape[0], -1])))


def split_channel(x):
    n = x.shape[1] // 2
    return x[:, :n], x[:, n:]


def get_graph_data(x, num_nodes, num_relations, num_features):
    """
    Converts a vector of shape [b, num_nodes, m] to Adjacency matrix
    of shape [b, num_relations, num_nodes, num_nodes]
    and a feature matrix of shape [b, num_nodes, num_features].
    :param x:
    :param num_nodes:
    :param num_relations:
    :param num_features:
    :return:
    """
    adj = F.reshape(x[:, :num_nodes*num_nodes*num_relations],
                    [-1, num_relations, num_nodes, num_nodes])
    feat_mat = F.reshape(x[:, num_nodes*num_nodes*num_relations:],
                         [-1, num_nodes, num_features])
    return adj, feat_mat


def Tensor2Mol(A, x):
    mol = Chem.RWMol()
    # x[x < 0] = 0.
    # A[A < 0] = -1
    # atoms_exist = np.sum(x, 1) != 0
    atoms = np.argmax(x, 1)
    atoms_exist = atoms != 4
    atoms = atoms[atoms_exist]
    atoms += 6
    adj = np.argmax(A, 0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    # print('num atoms: {}'.format(sum(atoms>0)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atom)))

    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])

    return mol


def construct_mol(x, A, atomic_num_list):
    mol = Chem.RWMol()
    # x (ch, num_node)
    atoms = np.argmax(x, axis=1)
    # last a
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]
    # print('num atoms: {}'.format(sum(atoms>0)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    # A (edge_type, num_node, num_node)
    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])

    return mol


def valid_mol(x):
    s = Chem.MolFromSmiles(Chem.MolToSmiles(x)) if x is not None else None
    if s is not None and '.' not in Chem.MolToSmiles(s):
        return s
    return None


def check_tensor(x):
    return valid_mol(Tensor2Mol(*x))


def adj_to_smiles(adj, x, atomic_num_list, gpu=-1):
    adj = _to_numpy_array(adj, gpu)
    x = _to_numpy_array(x, gpu)
    valid = [Chem.MolToSmiles(construct_mol(x_elem, adj_elem, atomic_num_list))
             for x_elem, adj_elem in zip(x, adj)]
    return valid


def check_validity(adj, x, atomic_num_list, gpu=-1, return_unique=True):
    adj = _to_numpy_array(adj, gpu)
    x = _to_numpy_array(x, gpu)
    valid = [valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list))
             for x_elem, adj_elem in zip(x, adj)]
    valid = [mol for mol in valid if mol is not None]
    print("valid molecules: {}/{}".format(len(valid), adj.shape[0]))
    for i, mol in enumerate(valid):
        print("[{}] {}".format(i, Chem.MolToSmiles(mol)))

    n_mols = x.shape[0]
    valid_ratio = len(valid)/n_mols
    valid_smiles = [Chem.MolToSmiles(mol) for mol in valid]
    unique_smiles = list(set(valid_smiles))
    unique_ratio = 0.
    if len(valid) > 0:
        unique_ratio = len(unique_smiles)/len(valid)
    if return_unique:
        valid_smiles = unique_smiles
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
    print("valid: {:.3f}%, unique: {:.3f}%".format(valid_ratio * 100, unique_ratio * 100))

    results = dict()
    results['valid_mols'] = valid_mols
    results['valid_smiles'] = valid_smiles
    results['valid_ratio'] = valid_ratio*100
    results['unique_ratio'] = unique_ratio*100

    return results


def check_novelty(gen_smiles, train_smiles):
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]
        novel = len(gen_smiles) - sum(duplicates)
        novel_ratio = novel*100./len(gen_smiles)
    print("novelty: {}%".format(novel_ratio))
    return novel_ratio


def _to_numpy_array(a, gpu=-1):
    if isinstance(a, chainer.Variable):
        a = a.array
    if gpu >= 0:
        return cuda.to_cpu(a)

    return a


def save_mol_png(mol, filepath, size=(600, 600)):
    Draw.MolToFile(mol, filepath, size=size)