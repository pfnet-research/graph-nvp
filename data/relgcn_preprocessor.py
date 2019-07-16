import numpy

from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor
import numpy as np


# --- Atom preprocessing ---
def construct_atomic_feature_matrix(mol, out_size=-1):
    """Returns atomic numbers of atoms consisting a molecule.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        out_size (int): The size of returned array.
            If this option is negative, it does not take any effect.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the tail of
            the array is padded with zeros.

    Returns:
        numpy.ndarray: an array consisting of atomic numbers
            of atoms in the molecule.
    """

    raise NotImplementedError


def construct_discrete_edge_matrix(mol, out_size=-1):
    """construct adjacency tensor. In addition to NxN adjacency matrix,
    third dimension stores one-hot encoded bond type.
    Args:
        mol (Chem.Mol):
        out_size (int):
    Returns (numpy.ndarray):
    """

    if mol is None:
        raise MolFeatureExtractionError('mol is None')
    N = mol.GetNumAtoms()

    if out_size < 0:
        size = N
    elif out_size >= N:
        size = out_size
    else:
        raise MolFeatureExtractionError('out_size {} is smaller than number '
                                        'of atoms in mol {}'
                                        .format(out_size, N))

    adjs = numpy.zeros((4, size, size), dtype=numpy.float32)
    for i in range(N):
        for j in range(N):
            bond = mol.GetBondBetweenAtoms(i, j)  # type: Chem.Bond
            if bond is not None:
                bond_type = str(bond.GetBondType())
                if bond_type == 'SINGLE':
                    adjs[1, i, j] = 1.0
                elif bond_type == 'DOUBLE':
                    adjs[2, i, j] = 1.0
                elif bond_type == 'TRIPLE':
                    adjs[3, i, j] = 1.0
            else:
                adjs[0,i,j] = 1.0
    return adjs


def one_hot(data, out_size):
    if out_size < 0:
        out_size = data.size
    b = np.zeros((out_size, 4))
    data = data[data>0]
    b[np.arange(data.size),data-6] = 1
    return b


def relGCN_pre(data, out_size):
    xd, adj_r = data
    x = one_hot(xd, out_size).astype(np.float32)
    adj = np.transpose(adj_r,(1,2,0)).astype(np.float32)
    return x, adj


class RELGCNPreprocessor(MolPreprocessor):
    """GGNN Preprocessor
    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.
    """

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False):
        super(RELGCNPreprocessor, self).__init__(add_Hs=add_Hs)
        if max_atoms >= 0 and 0 <= out_size < max_atoms:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size

    def get_input_features(self, mol):
        """get input features
        Args:
            mol (Mol):
        Returns:
        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        adj_tensor = construct_discrete_edge_matrix(mol, out_size=self.out_size)
        return relGCN_pre((atom_array, adj_tensor), out_size=self.out_size)
