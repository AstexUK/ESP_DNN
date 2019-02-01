from rdkit import Chem
from featurize import Featurize
from model_factory import custom_load_model
from data_processing import normalize
import numpy as np
import pickle
from rdkit.Chem import AllChem
import copy
import traceback
import os
from os.path import join


CHARGE_CORRECTION = 0.4
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

# (smarts, atom_indices, charge_correction)
EQUIVALENT_ATOMS = (
    # carboxylate
    ("[CX3](=O)[O-]", (1, 2), -CHARGE_CORRECTION),
    # ammonium
    ("[NX4+;!$(NC=O)]", (0,), CHARGE_CORRECTION),
    # Phosphates
    # TODO:
)


class AIChargeError(Exception):
    pass


def create_3d_mol(smiles, pdbfile):
    mol = Chem.MolFromSmiles(smiles)
    mol2 = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol2)
    Chem.MolToPDBFile(mol2, pdbfile)


class MolNeutralizer(object):

    def __init__(self, substitutions=[]):
        patts = [
            # Imidazoles
            ("[n+;H]", "n"),
            # Amines
            ("[N+;!H0]", "N"),
            # Carboxylic acids and alcohols
            ("[$([O-]);!$([O-][#7])]", "O"),
            # Thiols
            ("[S-;X1]", "S"),
            # Sulfonamides
            ("[$([N-;X2]S(=O)=O)]", "N"),
            # Enamines
            ("[$([N-;X2][C,N]=C)]", "N"),
            # Tetrazoles
            ("[n-]", "[nH]"),
            # Sulfoxides
            ("[$([S-]=O)]", "S"),
            # Amides
            ("[$([N-]C=O)]", "N"),
        ] + substitutions
        self.reactions = [
            (Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]

    def neutralize_smiles(self, smiles):
        return self.neutralize(Chem.MolFromSmiles(smiles))

    def neutralize(self, mol):
        for (reactant, product) in self.reactions:
            while mol.HasSubstructMatch(reactant):
                rms = AllChem.ReplaceSubstructs(mol, reactant, product)
                mol = rms[0]
        charge = sum([a.GetFormalCharge() for a in mol.GetAtoms()])
        return (mol, charge)


class MolChargePredictor(object):
    def __init__(self, model_file=join(SCRIPT_PATH, "model", "trained_model.h5"),
                 features_file=join(SCRIPT_PATH, "model", "feature_list.dat"),
                 norm_params_file=join(SCRIPT_PATH, "model", "norm_params.pkl")):
        self.model = custom_load_model(model_file)
        self.featurizer = Featurize(features_file=features_file, pad_value=0.0)
        self.skip_norm_mask = np.array([v.startswith("is_") for v in self.featurizer.features])
        with open(norm_params_file) as f:
            self.norm_params_dict = pickle.load(f)
        self.neutralizer = MolNeutralizer()
        self.equivalent_atoms = [(Chem.MolFromSmarts(ea[0]), ea[1], ea[2]) for ea in EQUIVALENT_ATOMS]

    @staticmethod
    def read_molecule_file(filepath, removeHs=True):
        if filepath.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(filepath, removeHs=removeHs)
        elif filepath.endswith(".mol"):
            mol = Chem.MolFromMolFile(filepath, removeHs=removeHs)
        else:
            raise AIChargeError("File format not understood")

        if mol is None:
            raise AIChargeError("Could not create molecule from file %s." % filepath)
        else:
            return mol

    def neutralize(self, mol):
        neutral_mol = copy.deepcopy(mol)
        for (fr, to) in self.neutralizer.reactions:
            for substruct in neutral_mol.GetSubstructMatches(fr):
                for aid in substruct:
                    neutral_mol.GetAtomWithIdx(aid).SetFormalCharge(0)

        if sum([a.GetFormalCharge() for a in neutral_mol.GetAtoms()]) != 0.:
            raise AIChargeError("Could not neutralize mol")
        neutral_smiles = Chem.MolToSmiles(neutral_mol)
        smiles_atom_order = eval(neutral_mol.GetProp("_smilesAtomOutputOrder"))
        atom_mappings = dict(zip(range(len(smiles_atom_order)), smiles_atom_order))
        neutral_mol = Chem.MolFromSmiles(neutral_smiles)
        return neutral_mol, atom_mappings

    def apply_charge_correction(self, mol):
        for (patt, aids, correction) in self.equivalent_atoms:
            for substruct in mol.GetSubstructMatches(patt):
                final_charge = (sum([mol.GetAtomWithIdx(substruct[aid]).GetPDBResidueInfo().GetOccupancy()
                                     for aid in aids]) + correction) / float(len(aids))
                for aid in aids:
                    mol.GetAtomWithIdx(substruct[aid]).GetPDBResidueInfo().SetOccupancy(final_charge)

    def predict_on_pdb_block(self, pdb_block):
        try:
            input_mol_with_Hs = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
            input_mol = neutral_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True)
            atom_mappings = dict(zip(range(input_mol.GetNumAtoms()), range(input_mol.GetNumAtoms())))

            if sum([a.GetFormalCharge() for a in input_mol.GetAtoms()]) != 0.:
                neutral_mol, atom_mappings = self.neutralize(input_mol)

            input_mol.Debug()
            neutral_mol.Debug()

            features_array, neighbours_array = self.featurizer.get_mol_fetaures(neutral_mol)
            features_array = np.expand_dims(features_array, axis=0)
            neighbours_array = np.expand_dims(neighbours_array, axis=0)
            features_array, _, _ = normalize(features_array, skip_norm_mask=self.skip_norm_mask, params_dict=self.norm_params_dict)

            charges = self.model.predict([features_array, neighbours_array]).flatten()
            charges = charges - (charges.sum() / len(charges))
            charges = charges.astype("float")

            for a in input_mol_with_Hs.GetAtoms():
                a.GetPDBResidueInfo().SetOccupancy(0.0)

            for (aid, cid) in atom_mappings.iteritems():
                # to write pqr
                input_mol_with_Hs.GetAtomWithIdx(aid).GetPDBResidueInfo().SetOccupancy(charges[cid])
            self.apply_charge_correction(input_mol_with_Hs)
            return Chem.MolToPDBBlock(input_mol_with_Hs)

        except:
            tb = traceback.format_exc()
            raise AIChargeError("Unhandled error occrued. Here is traceback:\n %s" % tb)

    def predict_on_pdb_file(self, pdb_file):
        with open(pdb_file) as f:
            return self.predict_on_pdb_block(f.read())
 