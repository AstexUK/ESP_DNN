#! /usr/bin/env python
#
# Copyright (C) 2015-2016 Rich Lewis <rl403@cam.ac.uk>
# License: 3-clause BSD
#
# Modified from https://github.com/richlewis42/scikit-chem/blob/master/skchem/features/atom.py
# by Prakash Chandra Rathi
#
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

import functools

import pandas as pd
import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors, rdPartialCharges
from rdkit.Chem.rdchem import HybridizationType

from os.path import join, dirname, abspath
from rdkit.Chem.AtomPairs.Utils import NumPiElectrons

SCRIPT_PATH = dirname(abspath(__file__))

PERIODIC_TABLE = pd.read_csv(
    join(SCRIPT_PATH, "data", "atom_data.csv"), index_col=0)
RD_PT = Chem.rdchem.GetPeriodicTable()


rd_log = RDLogger.logger()
rd_log.setLevel(RDLogger.CRITICAL)


def element(a):
    """ Return the element """

    return a.GetSymbol()


def is_element(a, symbol="C"):
    """ Is the atom of a given element """
    return element(a) == symbol


element_features = {"is_{}".format(e): functools.partial(is_element, symbol=e)
                    for e in ("B", "S", "C", "P", "O", "N", "I", "Cl", "F", "Br")}


def is_h_acceptor(a):
    """ Is an H acceptor? """

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return idx in [i[0] for i in Lipinski._HAcceptors(m)]


def is_h_donor(a):
    """ Is an H donor? """

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return idx in [i[0] for i in Lipinski._HDonors(m)]


def is_hetero(a):
    """ Is a heteroatom? """

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return idx in [i[0] for i in Lipinski._Heteroatoms(m)]


def atomic_number(a):
    """ Atomic number of atom """

    return a.GetAtomicNum()


def atomic_mass(a):
    """ Atomic mass of atom """

    return a.GetMass()


def explicit_valence(a):
    """ Explicit valence of atom """
    return a.GetExplicitValence()


def implicit_valence(a):
    """ Implicit valence of atom """

    return a.GetImplicitValence()


def valence(a):
    """ returns the valence of the atom """

    return explicit_valence(a) + implicit_valence(a)


def degree(a):
    """ returns the degree of the atom """

    return a.GetDegree()


def n_valence_electrons(a):
    """ return the number of valance electrons an atom has """

    return RD_PT.GetNOuterElecs(a.GetAtomicNum())


def n_pi_electrons(a):
    """ returns number of pi electrons """

    return NumPiElectrons(a)


def n_lone_pairs(a):
    """ returns the number of lone pairs assicitaed with the atom """

    return int(0.5 * (n_valence_electrons(a) - degree(a) - n_hydrogens(a) -
                      formal_charge(a) - n_pi_electrons(a)))


def van_der_waals_radius(a):
    """ returns van der waals radius of the atom """
    return PERIODIC_TABLE.van_der_waals_radius[a.GetAtomicNum()]


def formal_charge(a):
    """ Formal charge of atom """

    return a.GetFormalCharge()


def is_aromatic(a):
    """ Boolean if atom is aromatic"""

    return a.GetIsAromatic()


def num_implicit_hydrogens(a):
    """ Number of implicit hydrogens """

    return a.GetNumImplicitHs()


def num_explicit_hydrogens(a):
    """ Number of explicit hydrodgens """

    return a.GetNumExplicitHs()


def n_hydrogens(a):
    """ Number of hydrogens """

    return num_implicit_hydrogens(a) + num_explicit_hydrogens(a)


def is_in_ring(a):
    """ Whether the atom is in a ring """

    return a.IsInRing()


def crippen_log_p_contrib(a):
    """ Hacky way of getting logP contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return Crippen._GetAtomContribs(m)[idx][0]


def crippen_molar_refractivity_contrib(a):
    """ Hacky way of getting molar refractivity contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return Crippen._GetAtomContribs(m)[idx][1]


def tpsa_contrib(a):
    """ Hacky way of getting total polar surface area contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return rdMolDescriptors._CalcTPSAContribs(m)[idx]


def labute_asa_contrib(a):
    """ Hacky way of getting accessible surface area contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return rdMolDescriptors._CalcLabuteASAContribs(m)[0][idx]


def gasteiger_charge(a, force_calc=False):
    """ Hacky way of getting gasteiger charge """

    res = a.GetPropsAsDict().get("_GasteigerCharge", None)
    if res and not force_calc:
        return float(res)
    else:
        m = a.GetOwningMol()
        rdPartialCharges.ComputeGasteigerCharges(m)
        return float(a.GetProp("_GasteigerCharge"))


def pauling_electronegativity(a):
    return PERIODIC_TABLE.pauling_electronegativity[
        a.GetAtomicNum()]


def first_ionization(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), "first_ionisation_energy"]


def group(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), "group"]


def is_group(a, query_group=None):
    g = PERIODIC_TABLE.loc[a.GetAtomicNum(), "group"]
    if query_group is None:
        return np.isnan(g)
    else:
        return query_group == int(g)


group_features = {
    "is_group_{}".format(g): functools.partial(is_group, query_group=g)
    for g in range(1, 19) + [None]}


def period(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), "period"]


def is_period(a, query_period=None):
    p = PERIODIC_TABLE.loc[a.GetAtomicNum(), "period"]
    if query_period is None:
        return np.isnan(p)
    else:
        return query_period == int(p)


period_features = {
    "is_period_{}".format(p): functools.partial(is_period, query_period=p)
    for p in range(1, 8)}


def is_hybridized(a, hybrid_type=HybridizationType.SP3):
    """ Hybridized as type hybrid_type, default SP3 """

    return str(a.GetHybridization()) == str(hybrid_type)


hybridization_features = {"is_" + n + "_hybridized": functools.partial(
    is_hybridized, hybrid_type=n)
    for n in HybridizationType.names}

ATOM_FEATURES = {
    "atomic_number": atomic_number,
    "atomic_mass": atomic_mass,
    "formal_charge": formal_charge,
    "gasteiger_charge": gasteiger_charge,
    "pauling_electronegativity": pauling_electronegativity,
    "first_ionisation": first_ionization,
    "group": group,
    "period": period,
    "valence": valence,
    "degree": degree,
    "n_valence_electrons": n_valence_electrons,
    "n_pi_electrons": n_pi_electrons,
    "n_lone_pairs": n_lone_pairs,
    "van_der_waals_radius": van_der_waals_radius,
    "is_aromatic": is_aromatic,
    "n_hydrogens": n_hydrogens,
    "is_in_ring": is_in_ring,
    "log_p_contrib": crippen_log_p_contrib,
    "molar_refractivity_contrib": crippen_molar_refractivity_contrib,
    "is_h_acceptor": is_h_acceptor,
    "is_h_donor": is_h_donor,
    "is_heteroatom": is_hetero,
    "total_polar_surface_area_contrib": tpsa_contrib,
    "total_labute_accessible_surface_area": labute_asa_contrib,
}
ATOM_FEATURES.update(element_features)
ATOM_FEATURES.update(hybridization_features)
ATOM_FEATURES.update(group_features)
ATOM_FEATURES.update(period_features)
