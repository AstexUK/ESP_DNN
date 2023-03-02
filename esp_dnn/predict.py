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
import glob
import logging
import os
import subprocess

from rdkit import Chem
from esp_dnn.mol_charge_predictor import MolChargePredictor, \
    AIChargeError

log = logging.getLogger("predict" if __name__ == "__main__" else __name__)

os.environ["PLI_DIR"] = "pli"
pli_binary = os.path.join(os.environ["PLI_DIR"], "bin/pli")

def protein_pdb_file2pqr_file(protein_file):
    command_args = ["./"+pli_binary,
                    "-mode", "features",
                    "-protein", protein_file,
                    "-po", "pqr",
                    "-pn", protein_file]

    sp = subprocess.Popen(command_args,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    out, err = sp.communicate()
    if sp.returncode:
        raise AIChargeError(
            "PLI failed to create pqr file\nPLI Error:\n%s" % err)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("-m", "--mode",
                        help="Whether input files are ligands or proteins",
                        choices=["protein", "ligand"],
                        default="ligand")
    parser.add_argument("-i", "--input-dir",
                        help="Input directory containing PDB files")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--periodic-table", default="data/atom_data.csv")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(name)s:%(levelname)s] %(message)s")
    args = parse_args()
    mode = args.mode
    input_dir = args.input_dir
    model_dir = args.model_dir
    periodic_table = args.periodic_table

    log.info("Input dir is %s" % input_dir)

    # convert ligand mol files to pdb files
    if mode == "ligand":
        mcp = MolChargePredictor(model_dir=model_dir, periodic_table=periodic_table)
        for mol_file in glob.glob(os.path.join(input_dir, "*.mol")):
            try:
                log.info(mol_file)
                mol = Chem.MolFromMolFile(mol_file, removeHs=False)
                if mol is None:
                    log.warning("MOL2PDB: Error converting %s" % mol_file)
                    continue
                pdb_block = Chem.MolToPDBBlock(mol)
                mcp.pdb_block2pqr_file(pdb_block=pdb_block, output_file=mol_file+".pqr")
            except Exception as e:
                log.error("Error converting %s file" % mol_file)
                log.error(str(e))
    else:
        all_files = glob.glob(os.path.join(input_dir, "*.pdb"))
        for pdb_file in all_files:
            protein_pdb_file2pqr_file(protein_file=pdb_file)

