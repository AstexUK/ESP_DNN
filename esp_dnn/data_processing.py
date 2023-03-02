import numpy as np
import pickle

from rdkit import Chem
from rdkit.Chem import Mol, AllChem


def normalize(a, skip_norm_mask=None, params_pickle=None, params_dict=None):
    """ normalizez a 3d array (mol, atom, props) in prop axis """

    a_2d = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
    if params_pickle is not None:
        with open(params_pickle) as f:
            d = pickle.load(f)
            mean = d["mean"]
            std = d["std"]
    elif params_dict is not None:
        mean = params_dict["mean"]
        std = params_dict["std"]
    else:
        mean = np.nanmean(a_2d, axis=0, keepdims=True)
        std = np.nanmean(a_2d, axis=0, keepdims=True)

    if skip_norm_mask is not None:
        # do not standardized one-hot-encoded properties
        skip_norm_mask = np.expand_dims(skip_norm_mask, 0)
        # do not standardize all-zero props
        skip_norm_mask |= (std == 0.0)

        mean[skip_norm_mask] = 0.0
        std[skip_norm_mask] = 1.0

    return np.reshape((a_2d - mean) / std, a.shape), mean, std

def smiles_3DMol(smi:str) -> Mol:
  mol = Chem.AddHs(Chem.MolFromSmiles(smi))
  AllChem.EmbedMolecule(mol)
  AllChem.MMFFOptimizeMolecule(mol)
  return mol
