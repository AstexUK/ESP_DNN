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
import tensorflow as tf
from contextlib import contextmanager
import tempfile
import shutil
import subprocess
import logging


CHARGE_CORRECTION = 0.4
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger(__name__)

# (smarts, atom_indices, charge_correction)
EQUIVALENT_ATOMS = (
    # (smarts, tuple of atom ids in the smarts that are equivalent in terms of charge correction, charge correction)
    # carboxylate
    ("[CX3](=O)[O-]", (1, 2)),
    # ammonium
    ("[NX4+;!$(NC=O)]", (0,)),
    # amidinium
    ("[NX3][CX3]=[NX3+]", (0, 2)),
    # Nitro
    ("[N+](=O)[O-]", (1, 2)),
    # Terminal Phosphates with -2 charge
    ("[O&H0][PX4](=O)([OX1-])[OX1-]", (2, 3, 4)),
    # linking phosphates with -1 charge
    ("[O&H0][PX4](=O)([O-])[OX2&H0]", (2, 3)),
    # TODO: others?
)

PLI_DIR = "/home/golduser/src/pli2/stable"


def get_pli(pli_dir=PLI_DIR):
    if os.environ.get("PLI_DIR", None) != pli_dir:
        os.environ["PLI_DIR"] = pli_dir
    pli_exe = os.path.join(os.environ["PLI_DIR"], "bin/pli")
    return pli_exe



class AIChargeError(Exception):
    pass


def create_3d_mol(smiles, pdbfile):
    mol = Chem.MolFromSmiles(smiles)
    mol2 = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol2)
    Chem.MolToPDBFile(mol2, pdbfile)


@contextmanager
def TemporaryDirectory(cleanup=True):
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        if cleanup:
            shutil.rmtree(name)
        else:
            log.warn("Directory %s not removed" % name)


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
                 norm_params_file=join(SCRIPT_PATH, "model", "norm_params.pkl"),
                 debug=False,
                 clean_tmp_dir=True):
        # https://github.com/keras-team/keras/issues/5640
        self.graph = tf.get_default_graph()
        self.model = custom_load_model(model_file)
        self.model._make_predict_function()
        self.debug = debug
        self.clean_tmp_dir = clean_tmp_dir

        self.featurizer = Featurize(features_file=features_file, pad_value=0.0)
        self.skip_norm_mask = np.array([v.startswith("is_") for v in self.featurizer.features])
        with open(norm_params_file) as f:
            self.norm_params_dict = pickle.load(f)
        self.neutralizer = MolNeutralizer()
        self.equivalent_atoms = [(Chem.MolFromSmarts(ea[0]), ea[1]) for ea in EQUIVALENT_ATOMS]

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

    def apply_charge_correction(self, mol):
        # apply charge corrections for equivalent atoms (eg carboxylate oxygens).
        # Also add charges to charged groups (eq., carboxylates, amines, etc.)

        # account for formally charged atoms
        for (fr, to) in self.neutralizer.reactions:
            for substruct in mol.GetSubstructMatches(fr):
                for aid in substruct:
                    a = mol.GetAtomWithIdx(aid)
                    a.GetPDBResidueInfo().SetOccupancy(a.GetPDBResidueInfo().GetOccupancy() + CHARGE_CORRECTION * a.GetFormalCharge())

        for (patt, aids) in self.equivalent_atoms:
            for substruct in mol.GetSubstructMatches(patt):
                final_charge = sum([mol.GetAtomWithIdx(substruct[aid]).GetPDBResidueInfo().GetOccupancy()
                                    for aid in aids]) / float(len(aids))
                for aid in aids:
                    mol.GetAtomWithIdx(substruct[aid]).GetPDBResidueInfo().SetOccupancy(final_charge)

    def get_equivalent_atoms(self, mol):
        for (patt, aids, correction) in self.equivalent_atoms:
            for substruct in mol.GetSubstructMatches(patt):

                final_charge = (sum([mol.GetAtomWithIdx(substruct[aid]).GetPDBResidueInfo().GetOccupancy()
                                     for aid in aids]) + correction) / float(len(aids))
                for aid in aids:
                    mol.GetAtomWithIdx(substruct[aid]).GetPDBResidueInfo().SetOccupancy(final_charge)

    def get_dqs(self, mol_with_charges):

        dqs = [(atom.GetPDBResidueInfo().GetSerialNumber(),
                atom.GetPDBResidueInfo().GetOccupancy())
               for atom in mol_with_charges.GetAtoms() if atom.GetAtomicNum() != 1]

        return dqs

    def neutralize(self, mol):
        neutral_mol = copy.deepcopy(mol)
        for (fr, to) in self.neutralizer.reactions:
            for substruct in neutral_mol.GetSubstructMatches(fr):
                for aid in substruct:
                    a = neutral_mol.GetAtomWithIdx(aid)
                    orig_charge = a.GetFormalCharge()
                    a.SetFormalCharge(0)
                    # TODO: check
                    if orig_charge == 1 and a.GetNumExplicitHs():
                        a.SetNumExplicitHs(a.GetNumExplicitHs() - 1)
                    elif orig_charge == -1:
                        a.SetNumExplicitHs(a.GetNumExplicitHs() + 1)
        #Chem.SanitizeMol(neutral_mol, sanitizeOps=(Chem.SanitizeFlags.SANITIZE_ADJUSTHS))
        if sum([a.GetFormalCharge() for a in neutral_mol.GetAtoms()]) != 0.0:
            raise AIChargeError("Failed to neutralize molecule")
        Chem.SanitizeMol(neutral_mol)
        return neutral_mol

    def predict_dqs_from_pdb_block(self, pdb_block):
        try:
            input_mol_with_Hs = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
            input_mol = neutral_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True)

            # get a neutral mol and atom mapping between neutral and charged mols
            # We need a neutral mol because our model was trained on neutral mols
            if sum([a.GetFormalCharge() for a in input_mol.GetAtoms()]) != 0.:
                neutral_mol = self.neutralize(input_mol)

            # get features and neighbour matrices
            features_array, neighbours_array = self.featurizer.get_mol_fetaures(neutral_mol)
            features_array = np.expand_dims(features_array, axis=0)
            neighbours_array = np.expand_dims(neighbours_array, axis=0)
            features_array, _, _ = normalize(features_array, skip_norm_mask=self.skip_norm_mask, params_dict=self.norm_params_dict)

            # https://github.com/keras-team/keras/issues/5640
            # predict charges
            with self.graph.as_default():
                charges = self.model.predict([features_array, neighbours_array]).flatten()

            # Because we predict charge on each atom individually, they might not add up to zero.
            charges = charges.astype("float")
            charges = charges - (charges.sum() / len(charges))

            # initialise occupancies
            for a in input_mol_with_Hs.GetAtoms():
                a.GetPDBResidueInfo().SetOccupancy(0.0)

            # add charges to heavy atoms in the original molecule with Hs
            aid = 0
            for a in input_mol_with_Hs.GetAtoms():
                if a.GetAtomicNum() == 1:
                    continue
                a.GetPDBResidueInfo().SetOccupancy(charges[aid])
                aid += 1

            # apply charge corrections
            self.apply_charge_correction(input_mol_with_Hs)

            # get dqs
            return self.get_dqs(input_mol_with_Hs)

        except:
            tb = traceback.format_exc()
            raise AIChargeError("Unhandled error occrued. Here is the traceback:\n %s" % tb)

    def dqs2pqr(self, pdb_block, dqs):
        with TemporaryDirectory() as workdir:
            with open(join(workdir, "input.pdb"), "w") as f:
                f.write(pdb_block)
            command_args = [get_pli(), "-mode", "features",
                            "-ligand", join(workdir, "input.pdb"), "-ln", "output", "-lo", "pqr",
                            "-ldq", join(workdir, "dq.dat")]
            with open(join(workdir, "dq.dat"), "w") as f:
                f.write("\n".join(["%-6s %6d %6.3f" % ("dq", dq[0], dq[1])
                                   for dq in dqs]))
            sp = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=workdir)
            out, err = sp.communicate()
            if sp.returncode:
                raise AIChargeError("PLI failed to create pqr file")
            with open(join(workdir, "output.pqr")) as f:
                return f.read()

    def pdb_block2pqr(self, pdb_block):
        dqs = self.predict_on_pdb_block(pdb_block)
        return self.dqs2pqr(pdb_block, dqs)

    def predict_on_pdb_file(self, pdb_file):
        with open(pdb_file) as f:
            dqs = self.predict_on_pdb_block(f.read())


if __name__ == "__main__":
    mcp = MolChargePredictor()
    mcp.predict_on_pdb_file("/home/golduser/uwsgi/stable/apps/esp-ai-dev/esp_ai_lib/test/xiap_issue.pdb")
