from __future__ import absolute_import

from rdkit import Chem
from rdkit.Chem import AllChem


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
        return mol, charge
