from __future__ import absolute_import

import functools

import pandas
import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors, rdPartialCharges
from rdkit.Chem.rdchem import HybridizationType

from rdkit.Chem.AtomPairs.Utils import NumPiElectrons

rd_log = RDLogger.logger()
rd_log.setLevel(RDLogger.CRITICAL)

class AtomFeatures:
    def __init__(self, periodic_table):
        self.periodic_table = pandas.read_csv(periodic_table, index_col=0)
        self.rd_pt = Chem.rdchem.GetPeriodicTable()

    def element(self, a):
        """ Return the element """

        return a.GetSymbol()

    def is_element(self, a, symbol="C"):
        """ Is the atom of a given element """
        return self.element(a) == symbol

    def is_h_acceptor(self, a):
        """ Is an H acceptor? """

        m = a.GetOwningMol()
        idx = a.GetIdx()
        return idx in [i[0] for i in Lipinski._HAcceptors(m)]

    def is_h_donor(self, a):
        """ Is an H donor? """

        m = a.GetOwningMol()
        idx = a.GetIdx()
        return idx in [i[0] for i in Lipinski._HDonors(m)]

    def is_hetero(self, a):
        """ Is a heteroatom? """

        m = a.GetOwningMol()
        idx = a.GetIdx()
        return idx in [i[0] for i in Lipinski._Heteroatoms(m)]

    def atomic_number(self, a):
        """ Atomic number of atom """

        return a.GetAtomicNum()

    def atomic_mass(self, a):
        """ Atomic mass of atom """

        return a.GetMass()

    def explicit_valence(self, a):
        """ Explicit valence of atom """
        return a.GetExplicitValence()

    def implicit_valence(self, a):
        """ Implicit valence of atom """
        return a.GetImplicitValence()

    def valence(self, a):
        """ returns the valence of the atom """

        return self.explicit_valence(a) + self.implicit_valence(a)

    def degree(self, a):
        """ returns the degree of the atom """

        return a.GetDegree()

    def n_valence_electrons(self, a):
        """ return the number of valance electrons an atom has """

        return self.rd_pt.GetNOuterElecs(a.GetAtomicNum())

    def n_pi_electrons(self, a):
        """ returns number of pi electrons """

        return NumPiElectrons(a)

    def n_lone_pairs(self, a):
        """ returns the number of lone pairs assicitaed with the atom """

        return int(0.5 * (self.n_valence_electrons(a) - self.degree(a) - self.n_hydrogens(a) -
                          self.formal_charge(a) - self.n_pi_electrons(a)))

    def van_der_waals_radius(self, a):
        """ returns van der waals radius of the atom """
        return self.periodic_table.van_der_waals_radius[a.GetAtomicNum()]

    def formal_charge(self, a):
        """ Formal charge of atom """

        return a.GetFormalCharge()

    def is_aromatic(self, a):
        """ Boolean if atom is aromatic"""

        return a.GetIsAromatic()

    def num_implicit_hydrogens(self, a):
        """ Number of implicit hydrogens """

        return a.GetNumImplicitHs()

    def num_explicit_hydrogens(self, a):
        """ Number of explicit hydrodgens """

        return a.GetNumExplicitHs()

    def n_hydrogens(self, a):
        """ Number of hydrogens """

        return self.num_implicit_hydrogens(a) + self.num_explicit_hydrogens(a)

    def is_in_ring(self, a):
        """ Whether the atom is in a ring """

        return a.IsInRing()

    def crippen_log_p_contrib(self, a):
        """ Hacky way of getting logP contribution. """

        idx = a.GetIdx()
        m = a.GetOwningMol()
        return Crippen._GetAtomContribs(m)[idx][0]

    def crippen_molar_refractivity_contrib(self, a):
        """ Hacky way of getting molar refractivity contribution. """

        idx = a.GetIdx()
        m = a.GetOwningMol()
        return Crippen._GetAtomContribs(m)[idx][1]

    def tpsa_contrib(self, a):
        """ Hacky way of getting total polar surface area contribution. """

        idx = a.GetIdx()
        m = a.GetOwningMol()
        return rdMolDescriptors._CalcTPSAContribs(m)[idx]

    def labute_asa_contrib(self, a):
        """ Hacky way of getting accessible surface area contribution. """

        idx = a.GetIdx()
        m = a.GetOwningMol()
        return rdMolDescriptors._CalcLabuteASAContribs(m)[0][idx]

    def gasteiger_charge(self, a, force_calc=False):
        """ Hacky way of getting gasteiger charge """

        res = a.GetPropsAsDict().get("_GasteigerCharge", None)
        if res and not force_calc:
            return float(res)
        else:
            m = a.GetOwningMol()
            rdPartialCharges.ComputeGasteigerCharges(m)
            return float(a.GetProp("_GasteigerCharge"))

    def pauling_electronegativity(self, a):
        return self.periodic_table.pauling_electronegativity[
            a.GetAtomicNum()]

    def first_ionization(self, a):

        return self.periodic_table.loc[a.GetAtomicNum(), "first_ionisation_energy"]

    def group(self, a):

        return self.periodic_table.loc[a.GetAtomicNum(), "group"]

    def is_group(self, a, query_group=None):
        g = self.periodic_table.loc[a.GetAtomicNum(), "group"]
        if query_group is None:
            return np.isnan(g)
        else:
            return query_group == int(g)



    def period(self, a):
        return self.periodic_table.loc[a.GetAtomicNum(), "period"]

    def is_period(self, a, query_period=None):
        p = self.periodic_table.loc[a.GetAtomicNum(), "period"]
        if query_period is None:
            return np.isnan(p)
        else:
            return query_period == int(p)
    def is_hybridized(self, a, hybrid_type=HybridizationType.SP3):
        """ Hybridized as type hybrid_type, default SP3 """

        return str(a.GetHybridization()) == str(hybrid_type)

    def get_atom_features(self):
        ATOM_FEATURES = {
            "atomic_number": self.atomic_number,
            "atomic_mass": self.atomic_mass,
            "formal_charge": self.formal_charge,
            "gasteiger_charge": self.gasteiger_charge,
            "pauling_electronegativity": self.pauling_electronegativity,
            "first_ionisation": self.first_ionization,
            "group": self.group,
            "period": self.period,
            "valence": self.valence,
            "degree": self.degree,
            "n_valence_electrons": self.n_valence_electrons,
            "n_pi_electrons": self.n_pi_electrons,
            "n_lone_pairs": self.n_lone_pairs,
            "van_der_waals_radius": self.van_der_waals_radius,
            "is_aromatic": self.is_aromatic,
            "n_hydrogens": self.n_hydrogens,
            "is_in_ring": self.is_in_ring,
            "log_p_contrib": self.crippen_log_p_contrib,
            "molar_refractivity_contrib": self.crippen_molar_refractivity_contrib,
            "is_h_acceptor": self.is_h_acceptor,
            "is_h_donor": self.is_h_donor,
            "is_heteroatom": self.is_hetero,
            "total_polar_surface_area_contrib": self.tpsa_contrib,
            "total_labute_accessible_surface_area": self.labute_asa_contrib,
        }

        element_features = {"is_{}".format(e): functools.partial(self.is_element, symbol=e)
                            for e in
                            ("B", "S", "C", "P", "O", "N", "I", "Cl", "F", "Br")}
        period_features = {
            "is_period_{}".format(p): functools.partial(self.is_period, query_period=p)
            for p in range(1, 8)}
        group_features = {
            "is_group_{}".format(g): functools.partial(self.is_group, query_group=g)
            for g in list(range(1, 19)) + [None]}
        hybridization_features = {"is_" + n + "_hybridized": functools.partial(
            self.is_hybridized, hybrid_type=n)
            for n in HybridizationType.names}
        ATOM_FEATURES.update(element_features)
        ATOM_FEATURES.update(hybridization_features)
        ATOM_FEATURES.update(group_features)
        ATOM_FEATURES.update(period_features)
        return ATOM_FEATURES
