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
import logging
import os

import numpy as np
from rdkit import Chem
import xarray as xr

from .atom_features import ATOM_FEATURES

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger(__name__)


class Featurize(object):
    def __init__(self, smiles_file=None,
                 id_smiles=None,
                 features_file=os.path.join(
                     SCRIPT_PATH, "data", "feature_list.dat"),
                 max_hac=None,
                 pad_value=np.nan):
        self.smiles_file = smiles_file
        self.id_smiles = id_smiles
        self.features_file = features_file
        self.max_hac = max_hac
        self.features = self.__get_feature_list()
        self.n_features = len(self.features)
        self.pad_value = pad_value

    def process(self):
        if self.id_smiles is None:
            self.id_smiles = self.read_smiles_file()
        self.ds = self.__featurize_smiles(self.id_smiles)
        return self.ds

    def export(self, filename):
        self.ds.to_netcdf(filename)

    def __get_feature_list(self):
        return sorted(f for f in (
            l.strip() for l in open(self.features_file).readlines())
            if f)

    def get_mol_fetaures(self, mol):
        n_atoms = self.max_hac or mol.GetNumAtoms()
        assert n_atoms >= mol.GetNumAtoms()
        feature_array = np.empty(shape=(n_atoms, self.n_features))
        feature_array[:] = self.pad_value
        neighbor_array = np.zeros(shape=(n_atoms, n_atoms))

        for atom in mol.GetAtoms():
            atom_id = atom.GetIdx()
            for feature_id, f in enumerate(self.features):
                feature_array[atom_id, feature_id] = ATOM_FEATURES[f](atom)
            for neigh_id in [n.GetIdx() for n in atom.GetNeighbors()]:
                neighbor_array[atom_id][neigh_id] = 1.0

        return feature_array, neighbor_array

    def read_smiles_file(self):
        """ read comma-separated smiles file (ID in the first column and smiles
        in the second) """

        id_smiles = []
        with open(self.smiles_file) as f:
            for line in f:
                tokens = line.strip().split(",")
                if len(tokens) != 2:
                    log.warn("Malformed line: %s" % line)
                    continue
                id_smiles.append((tokens[0].strip(), tokens[1].strip()))
        return id_smiles

    def __featurize_smiles(self, id_smiles):
        valid_ids = []
        invalid_ids = []
        valid_smiles = []
        feature_arrays = []
        neighbor_arrays = []
        for id_, smiles in id_smiles:
            mol = Chem.MolFromSmiles(smiles, )
            if mol is None:
                continue
            try:
                fa, na = self.get_mol_fetaures(mol)
                feature_arrays.append(fa)
                neighbor_arrays.append(na)
                valid_ids.append(id_)
                valid_smiles.append(smiles)
            except AssertionError:
                invalid_ids.append(id_)
                continue

        ds = xr.Dataset(
            {"X": (("mol_id", "atom_id", "feature"), np.stack(feature_arrays)),
             "D": (("mol_id", "atom_id", "atom_id"), np.stack(neighbor_arrays)),
             "SMILES": (("mol_id",), list(valid_smiles))},
            coords={"mol_id": valid_ids,
                    "atom_id": range(self.max_hac),
                    "feature": self.features
                    })

        if len(invalid_ids):
            log.warn("%d invalid smiles found" % len(invalid_ids))

        return ds


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file",
                        help="Input smiles file (a comma-separated file "
                        "containing molecule id and smiles)")
    parser.add_argument("-f", "--features_file",
                        help="file containing list of features",
                        default=os.path.join(SCRIPT_PATH, "feature_list.dat"))
    parser.add_argument("--max_hac",
                        help="Maximum heavy atom count. feature matrices will "
                        "have these many atoms", type=int)
    parser.add_argument("-o", "--output_file",
                        help="Input smiles file (a comma-separated file "
                        "containing molecule id and smiles)")
    args = parser.parse_args()
    f = Featurize(args.input_file,
                  features_file=args.features_file, max_hac=args.max_hac)
    f.process()
    f.export(args.output_file)


if __name__ == "__main__":
    main()
