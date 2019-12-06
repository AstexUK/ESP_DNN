# Copyright 2019 Astex Therapeutics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

from argparse import ArgumentParser
from contextlib import contextmanager
import copy
import glob
import logging
import numpy as np
import os
import pickle
import shutil
import subprocess
import tempfile
import traceback

from rdkit import Chem
from rdkit.Chem import AllChem

from .data_processing import normalize
from .featurize import Featurize


CHARGE_CORRECTION = 0.4
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger("predict" if __name__ == "__main__" else __name__)

EQUIVALENT_ATOMS = (
    # (smarts, tuple of atom ids in the smarts that are equivalent in terms
    # of charge correction, charge correction)
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


class AIChargeError(Exception):
    pass


def get_pli():
    pli_dir = os.environ.get("PLI_DIR")
    if pli_dir is None:
        pli_dir = os.path.join(SCRIPT_PATH, "ext", "pli")
        if os.path.exists(os.path.join(pli_dir, "bin", "pli")):
            os.environ["PLI_DIR"] = pli_dir
        else:
            raise AIChargeError(
                "Neither PLI_DIR environment variable is defined "
                "nor PLI executable could be found in %s" % pli_dir)
    return os.path.join(os.environ["PLI_DIR"], "bin/pli")


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
            (Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False))
            for x, y in patts]

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
    def __init__(self, model_file=os.path.join(SCRIPT_PATH, "model",
                                               "trained_model.h5"),
                 features_file=os.path.join(
                     SCRIPT_PATH, "model", "feature_list.dat"),
                 norm_params_file=os.path.join(
                     SCRIPT_PATH, "model", "norm_params.pkl"),
                 debug=False,
                 clean_tmp_dir=True):
        # defer import until needed
        import tensorflow as tf
        from .model_factory import custom_load_model

        self.graph = tf.get_default_graph()
        self.model = custom_load_model(model_file)
        self.model._make_predict_function()
        self.debug = debug
        self.clean_tmp_dir = clean_tmp_dir

        self.featurizer = Featurize(features_file=features_file, pad_value=0.0)
        self.skip_norm_mask = np.array(
            [v.startswith("is_") for v in self.featurizer.features])
        with open(norm_params_file) as f:
            self.norm_params_dict = pickle.load(f)
        self.neutralizer = MolNeutralizer()
        self.equivalent_atoms = [(Chem.MolFromSmarts(ea[0]), ea[1])
                                 for ea in EQUIVALENT_ATOMS]

    @staticmethod
    def read_molecule_file(filepath, removeHs=True):
        if filepath.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(filepath, removeHs=removeHs)
        elif filepath.endswith(".mol"):
            mol = Chem.MolFromMolFile(filepath, removeHs=removeHs)
        else:
            raise AIChargeError("File format not understood")

        if mol is None:
            raise AIChargeError(
                "Could not create molecule from file %s." % filepath)
        else:
            return mol

    def apply_charge_correction(self, mol):
        """apply charge corrections for equivalent atoms (eg carboxylate oxygens).
        Also add charges to charged groups (eq., carboxylates, amines, etc.)"""
        for (fr, _) in self.neutralizer.reactions:
            for substruct in mol.GetSubstructMatches(fr):
                for aid in substruct:
                    a = mol.GetAtomWithIdx(aid)
                    a.GetPDBResidueInfo().SetOccupancy(
                        a.GetPDBResidueInfo()
                        .GetOccupancy() + CHARGE_CORRECTION *
                        a.GetFormalCharge())

        for (patt, aids) in self.equivalent_atoms:
            for substruct in mol.GetSubstructMatches(patt):
                final_charge = sum([mol.GetAtomWithIdx(substruct[aid])
                                       .GetPDBResidueInfo()
                                       .GetOccupancy()
                                    for aid in aids]) / float(len(aids))
                for aid in aids:
                    mol.GetAtomWithIdx(substruct[aid]).GetPDBResidueInfo(
                    ).SetOccupancy(final_charge)

    def get_equivalent_atoms(self, mol):
        for (patt, aids, correction) in self.equivalent_atoms:
            for substruct in mol.GetSubstructMatches(patt):

                final_charge = sum([mol.GetAtomWithIdx(substruct[aid])
                                    .GetPDBResidueInfo()
                                    .GetOccupancy()
                                    for aid in aids]) + correction
                final_charge /= float(len(aids))
                for aid in aids:
                    (mol.GetAtomWithIdx(substruct[aid])
                        .GetPDBResidueInfo()
                        .SetOccupancy(final_charge))

    def get_dqs(self, mol_with_charges):

        dqs = [(atom.GetPDBResidueInfo().GetSerialNumber(),
                atom.GetPDBResidueInfo().GetOccupancy())
               for atom in mol_with_charges.GetAtoms()
               if atom.GetAtomicNum() != 1]

        return dqs

    def neutralize(self, mol):
        neutral_mol = copy.deepcopy(mol)
        for (fr, _) in self.neutralizer.reactions:
            for substruct in neutral_mol.GetSubstructMatches(fr):
                for aid in substruct:
                    a = neutral_mol.GetAtomWithIdx(aid)
                    orig_charge = a.GetFormalCharge()
                    a.SetFormalCharge(0)
                    if orig_charge == 1 and a.GetNumExplicitHs():
                        a.SetNumExplicitHs(a.GetNumExplicitHs() - 1)
                    elif orig_charge == -1:
                        a.SetNumExplicitHs(a.GetNumExplicitHs() + 1)
        if sum([a.GetFormalCharge() for a in neutral_mol.GetAtoms()]) != 0.0:
            raise AIChargeError("Failed to neutralize molecule")
        Chem.SanitizeMol(neutral_mol)
        return neutral_mol

    def predict_dqs_from_pdb_block(self, pdb_block):
        try:
            input_mol_with_Hs = Chem.MolFromPDBBlock(
                pdb_block, removeHs=False, proximityBonding=False)
            input_mol = neutral_mol = Chem.MolFromPDBBlock(
                pdb_block, removeHs=True, proximityBonding=False)

            if input_mol_with_Hs is None or input_mol is None:
                raise AIChargeError(
                    "Failed to read molecule from the PDB block")

            # We need a neutral mol because our model was trained on
            # neutral mols
            if sum(a.GetFormalCharge() for a in input_mol.GetAtoms()) != 0.:
                neutral_mol = self.neutralize(input_mol)

            # get features and neighbour matrices
            features_array, neighbours_array = (
                self.featurizer.get_mol_fetaures(neutral_mol))

            features_array = np.expand_dims(features_array, axis=0)
            neighbours_array = np.expand_dims(neighbours_array, axis=0)
            features_array, _, _ = normalize(
                features_array, skip_norm_mask=self.skip_norm_mask,
                params_dict=self.norm_params_dict)

            # https://github.com/keras-team/keras/issues/5640
            # predict charges
            with self.graph.as_default():
                charges = self.model.predict(
                    [features_array, neighbours_array]).flatten()

            # Because we predict charge on each atom individually,
            # they might not add up to zero.
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

        except Exception:
            tb = traceback.format_exc()
            raise AIChargeError(
                "Unhandled error occrued. Here is the traceback:\n %s" % tb)

    def dqs2pqr(self, pdb_block, dqs):
        with TemporaryDirectory() as workdir:
            input_pdb_file = os.path.join(workdir, "input.pdb")
            input_dq_file = os.path.join(workdir, "dq.dat")

            # write data to files for PLI
            with open(input_pdb_file, "w") as f:
                f.write(pdb_block)
            with open(input_dq_file, "w") as f:
                f.write("\n".join(["%-6s %6d %6.3f" % ("dq", dq[0], dq[1])
                                   for dq in dqs]))

            # setup and pli command
            command_args = [get_pli(),
                            "-mode", "features",
                            "-ligand", input_pdb_file,
                            "-ln", "output",
                            "-lo", "pqr",
                            "-ldq", input_dq_file]
            sp = subprocess.Popen(command_args,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  cwd=workdir)
            _, err = sp.communicate()
            if sp.returncode:
                raise AIChargeError(
                    "PLI failed to create pqr file\nPLI Error:\n%s" % err)
            with open(os.path.join(workdir, "output.pqr")) as f:
                return f.read()

    def pdb_block2pqr_block(self, pdb_block):
        dqs = self.predict_dqs_from_pdb_block(pdb_block)
        return self.dqs2pqr(pdb_block, dqs)

    def pdb_file2pqr_block(self, pdb_file):
        with open(pdb_file) as f:
            return self.pdb_block2pqr_block(f.read())

    def pdb_file2pqr_file(self, pdb_file, pqr_file):
        with open(pqr_file, "w") as f:
            f.write(self.pdb_file2pqr_block(pdb_file))


def protein_pdb_file2pqr_file(protein_file, output_file):
    command_args = [get_pli(),
                    "-mode", "features",
                    "-protein", protein_file,
                    "-po", "pqr",
                    "-pn", "output_protein"]

    with TemporaryDirectory() as workdir:
        sp = subprocess.Popen(command_args,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              cwd=workdir)
        out, err = sp.communicate()
        tmp_output_file = os.path.join(workdir, "output_protein.pqr")
        if not sp.returncode and os.path.exists(tmp_output_file):
            return shutil.copyfile(tmp_output_file, output_file)
        else:
            raise AIChargeError("Failed to generate pqr file.\n\n"
                                "PLI stdout:\n%sPLI stderr:\n%s" % (out, err))


def pli_molfile2pdbblock(input_file):
    command_args = [get_pli(), "-mode", "preplig", "-ligand",
                    input_file, "-ln", "output", "-lo", "pdb"]
    with TemporaryDirectory() as workdir:
        sp = subprocess.Popen(
            command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=workdir)
        out, err = sp.communicate()
        if not sp.returncode:
            with open(os.path.join(workdir, "output.pdb")) as f:
                return f.read()
        log.warn((out, err))
        return None


def run(mode, input_dir, output_dir, stop_on_error):
    log.info("Ruuning in %s mode" % mode)
    if not os.path.isdir(input_dir):
        raise ValueError("Input directory (%s) does not exist" % input_dir)
    input_dir = os.path.abspath(input_dir)
    if output_dir is None:
        output_dir = input_dir
    log.info("Input dir is %s" % input_dir)
    log.info("Output dir is %s" % output_dir)

    # convert ligand mol files to pdb files
    if mode == "ligand":
        for mol_file in glob.glob(os.path.join(input_dir, "*.mol")):
            mol = Chem.MolFromMolFile(mol_file, removeHs=False)
            if mol is None:
                log.warn("MOL2PDB: Error converting %s" % mol_file)
                continue
            pdb_file = os.path.join(output_dir, os.path.basename(mol_file) + ".pdb")
            log.info("MOL2PDB: %s -> %s" % (mol_file, pdb_file))
            Chem.MolToPDBFile(mol, pdb_file)

    all_files = glob.glob(os.path.join(input_dir, "*.pdb"))

    if not len(all_files):
        log.error("No input files found in %s directory" % input_dir)
        return
    if mode == "ligand":
        log.info("Loading the trained model from %s directory" %
                 os.path.join(SCRIPT_PATH, "model"))
        mcp = MolChargePredictor()

    for pdb_file in all_files:
        pqr_file = os.path.join(
            output_dir, os.path.basename(pdb_file) + ".pqr")
        log.info("%s -> %s" % (pdb_file, pqr_file))
        try:
            if mode == "ligand":
                mcp.pdb_file2pqr_file(pdb_file, pqr_file)
            else:
                protein_pdb_file2pqr_file(pdb_file, pqr_file)
        except AIChargeError, e:
            if stop_on_error:
                raise
            else:
                log.error("Error converting %s file" % pdb_file)
                log.error(str(e))


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("-m", "--mode",
                        help="Whether input files are ligands or proteins",
                        choices=["protein", "ligand"],
                        default="ligand")
    parser.add_argument("-i", "--input_dir",
                        help="Input directory containing PDB files",
                        default=os.getcwd())
    parser.add_argument("-o", "--output_dir",
                        help="Output directory in which PQR files are written."
                        " Output files are written to the same directory if "
                        "not specified.")
    parser.add_argument("-e", "--stop_on_error",
                        help="Stop processing remaining files if an error "
                        "occurs.",
                        action="store_true",
                        default=False)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(name)s:%(levelname)s] %(message)s")
    args = parse_args()
    run(args.mode, args.input_dir, args.output_dir, args.stop_on_error)
