from argparse import ArgumentParser
import logging
import os

from rdkit import Chem

from esp_dnn.config import PERIODIC_TABLE
from esp_dnn.data_processing import smiles_3DMol
from esp_dnn.mol_charge_predictor import MolChargePredictor

logging.basicConfig(level=logging.INFO,
                    format="[%(name)s:%(levelname)s] %(message)s")

os.environ["PLI_DIR"] = "pli"
pli_binary = os.path.join(os.environ["PLI_DIR"], "bin/pli")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(name)s:%(levelname)s] %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--smiles")
    parser.add_argument("--mol-file")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    if args.smiles:
        mol = smiles_3DMol(args.smiles)
    elif args.mol_file:
        mol = Chem.MolFromMolFile(args.mol_file, removeHs=False)
    else:
        raise ValueError("You need to specify either a --smiles or a --mol-file parameter as your molecule input")
    pdb_block = Chem.MolToPDBBlock(mol)
    mcp = MolChargePredictor(model_dir=args.model_dir, periodic_table=PERIODIC_TABLE)
    mcp.pdb_block2pqr_file(pdb_block=pdb_block, output_file=args.output_file)
