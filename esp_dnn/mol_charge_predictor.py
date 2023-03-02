from __future__ import absolute_import

import copy
import logging
import os
import pickle
import shutil
import subprocess
import tempfile
import traceback
from contextlib import contextmanager
import numpy as np
import tensorflow as tf
from rdkit import Chem

from esp_dnn.data_processing import normalize
from esp_dnn.featurize import Featurize
from esp_dnn.model_factory import custom_load_model
from esp_dnn.mol_neutralizer import MolNeutralizer

CHARGE_CORRECTION = 0.4
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

@contextmanager
def TemporaryDirectory(cleanup=True):
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        if cleanup:
            shutil.rmtree(name)
        else:
            logging.warning("Directory %s not removed" % name)

class AIChargeError(Exception):
    pass


class MolChargePredictor(object):
    def __init__(self, model_dir, periodic_table,
                 debug=False,
                 clean_tmp_dir=True):

        self.model_file = os.path.join(model_dir, "trained_model.h5")
        features_file = os.path.join(model_dir, "feature_list.dat")
        norm_params_file = os.path.join(model_dir, "norm_params.pkl")

        self.graph = tf.compat.v1.get_default_graph()
        with self.graph.as_default():
            self.model = custom_load_model(self.model_file)
        self.debug = debug
        self.clean_tmp_dir = clean_tmp_dir

        self.featurizer = Featurize(features_file=features_file, pad_value=0.0, periodic_table=periodic_table)
        self.skip_norm_mask = np.array(
            [v.startswith("is_") for v in self.featurizer.features])
        with open(norm_params_file, "rb") as f:
            self.norm_params_dict = pickle.load(f, encoding="latin1")
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

    def predict_charges(self, features_array, neighbours_array):
        # https://github.com/keras-team/keras/issues/5640
        # predict charges
        with self.graph.as_default():
            charges = self.model.predict(
                [features_array, neighbours_array]).flatten()
        # Because we predict charge on each atom individually,
        # they might not add up to zero.
        charges = charges.astype("float")
        return charges - (charges.sum() / len(charges))

    def featurize(self, pdb_block):
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
        return features_array, neighbours_array, input_mol_with_Hs

    def predict_dqs_from_pdb_block(self, pdb_block):
        try:
            features_array, neighbours_array, input_mol_with_Hs = self.featurize(pdb_block)
            charges = self.predict_charges(features_array=features_array, neighbours_array=neighbours_array)
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
            command_args = ["./pli/bin/pli",
                            "-mode", "features",
                            "-ligand", input_pdb_file,
                            "-ln", input_pdb_file,
                            "-lo", "pqr",
                            "-ldq", input_dq_file]
            sp = subprocess.Popen(command_args,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  # cwd=output_dir,
                                  # shell=True
                                  )
            _, err = sp.communicate()
            if sp.returncode:
                raise AIChargeError(
                    "PLI failed to create pqr file\nPLI Error:\n%s" % err)
            with open(input_pdb_file+".pqr") as f:
                return f.read()

    def pdb_block2pqr_file(self, pdb_block, output_file=None):
        dqs = self.predict_dqs_from_pdb_block(pdb_block)
        dqr_block =  self.dqs2pqr(pdb_block=pdb_block, dqs=dqs)
        if output_file:
            with open(output_file, "w") as f:
                f.write(dqr_block)
        return dqr_block


