import logging
from datetime import datetime
from statistics import mean

import numpy
import xarray

from esp_dnn.config import PERIODIC_TABLE
from rdkit import Chem
from sklearn.metrics import mean_squared_error

from esp_dnn.data_processing import smiles_3DMol, normalize
from esp_dnn.mol_charge_predictor import MolChargePredictor

logger = logging.getLogger()

def featurize_smiles(smiles, mcp):
    mol = smiles_3DMol(smiles)
    hac = mol.GetNumHeavyAtoms()
    pdb_block = Chem.MolToPDBBlock(mol)
    features_array, neighbours_array, input_mol_with_Hs = mcp.featurize(pdb_block)
    return features_array, neighbours_array, hac


def featurize(mcp, features_array):
    features_array, _, _ = normalize(
        features_array, skip_norm_mask=mcp.skip_norm_mask,
        params_dict=mcp.norm_params_dict)
    return features_array

def evaluate(ds_disk, mcp):
    training_rms = []
    dev_rms = []
    test_rms = []
    molecules_with_nans_in_between = 0
    for i in range(len(ds_disk["X"].values)):
        features_array = numpy.expand_dims(ds_disk["X"].values[i], axis=0)
        features_array[numpy.isnan(features_array)] = 0
        features_array = featurize(mcp=mcp, features_array=features_array)
        neighbours_array = numpy.expand_dims(ds_disk["D"].values[i], axis=0)
        charges = ds_disk["Y"].values[i]
        smiles = ds_disk["SMILES"].values[i].decode()
        try:
            mol = smiles_3DMol(smiles)
            hac = mol.GetNumHeavyAtoms()
            charges = charges[:hac]
            charges = [yi[0] for yi in charges]
            if numpy.isnan(charges).any():
                molecules_with_nans_in_between += 1
                continue
            with mcp.graph.as_default():
                predicted_charges = mcp.model.predict(
                    [features_array, neighbours_array]).flatten()[:hac]
            rms = mean_squared_error(charges, predicted_charges.tolist(), squared=False)
            if ds_disk["train"].values[i]:
                training_rms.append(rms)
            elif ds_disk["valid"].values[i]:
                dev_rms.append(rms)
            else:
                test_rms.append(rms)
        except Exception as e:
            print(e)
        if i % 10000 == 0:
            logger.info("%s %s %s"%(datetime.now(), i, molecules_with_nans_in_between))
    logger.info("Mean RMSE in training set: %s %s"% (mean(training_rms),len(training_rms)))
    logger.info("Mean RMSE in dev set: %s %s"%(mean(dev_rms), len(dev_rms)))
    logger.info("Mean RMSE in test set: %s %s"%(mean(test_rms), len(test_rms)))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="[%(name)s:%(levelname)s] %(message)s")
    mcp = MolChargePredictor(model_dir="model", periodic_table=PERIODIC_TABLE)

    logger.info("Chembl")
    ds_disk = xarray.open_dataset("data/chembl.nc")
    evaluate(ds_disk, mcp=mcp)

    logger.info("Training/dev/test set")
    ds_disk = xarray.open_dataset("data/emolecules.nc")
    evaluate(ds_disk, mcp=mcp)

