from rootpy.tree import Cut
from math import pi

from ..base import Category
from ... import MMC_MASS
# All basic cut definitions are here


# Tau Selection
TAU_CHARGE = Cut('abs(ditau_tau0_q) == 1 && abs(ditau_tau1_q) == 1')
TAU_NTRACKS = Cut('(ditau_tau0_n_tracks == 1 || ditau_tau0_n_tracks == 3) && (ditau_tau1_n_tracks == 1 || ditau_tau1_n_tracks == 3)')
TAU_ID = Cut('ditau_tau0_jet_bdt_medium == 1 && ditau_tau1_jet_bdt_medium == 1')
TAU_SELECTION = (
        TAU_CHARGE
        & TAU_NTRACKS
        & TAU_ID
        )

# Di-Tau Selection
LEADING = Cut('ditau_tau0_pt > 40')
SUBLEADING = Cut('ditau_tau1_pt > 30')
OS = Cut('selection_opposite_sign == 1')
MET = Cut('selection_met == 1')
DETA = Cut('selection_delta_eta == 1')
DR = Cut('selection_delta_r == 1')

DITAU_SELECTION = (
        TAU_CHARGE
        & TAU_NTRACKS
        & TAU_ID
        & LEADING
        & SUBLEADING
        & OS
        & MET
        & DETA
        & DR
        )

class Category_TauSelection(Category):
    name = 'tau_selection'
    label = '#tau_{had}#tau_{had} Tau Preselection'
    common_cuts = TAU_SELECTION

class Category_DiTauSelection(Category):
    name = 'ditau_selection'
    label = '#tau_{had}#tau_{had} Di-Tau Preselection'
    common_cuts = DITAU_SELECTION

TAU_SAME_VERTEX = Cut('tau_same_vertex')

LEAD_TAU_40 = Cut('ditau_tau0_pt > 40')
SUBLEAD_TAU_30 = Cut('ditau_tau1_pt > 30')

LEAD_JET_50 = Cut('jet_0_pt > 50')
SUBLEAD_JET_30 = Cut('jet_1_pt > 30')
AT_LEAST_1JET = Cut('jet_0_pt > 30')

CUTS_2J = LEAD_JET_50 & SUBLEAD_JET_30
CUTS_1J = LEAD_JET_50 & (- SUBLEAD_JET_30)
CUTS_0J = (- LEAD_JET_50)

MET = Cut('met_et > 20')
DR_TAUS = Cut('0.8 < ditau_dr < 2.4')
DETA_TAUS = Cut('ditau_deta < 1.5')
DETA_TAUS_CR = Cut('ditau_deta > 1.5')
RESONANCE_PT = Cut('ditau_vect_sum_pt > 100')

# use .format() to set centality value
MET_CENTRALITY = 'ditau_met_bisect==1 || (ditau_met_min_dphi < {0})'

# common preselection cuts
PRESELECTION = (
    LEAD_TAU_40
    & SUBLEAD_TAU_30
    # & ID_MEDIUM # implemented in regions
    & MET
    & Cut('%s > 0' % MMC_MASS)
    & DR_TAUS
    # & TAU_SAME_VERTEX
    )

# VBF category cuts
CUTS_VBF = (
    CUTS_2J
    & DETA_TAUS
    )

CUTS_VBF_CR = (
    CUTS_2J
    & DETA_TAUS_CR
    )

# Boosted category cuts
CUTS_BOOSTED = (
    RESONANCE_PT
    & DETA_TAUS
    )

CUTS_BOOSTED_CR = (
    RESONANCE_PT
    & DETA_TAUS_CR
    )


class Category_Preselection_NO_MET_CENTRALITY(Category):
    name = 'preselection'
    label = '#tau_{had}#tau_{had} Preselection'
    common_cuts = PRESELECTION


class Category_Preselection(Category):
    name = 'preselection'
    label = '#tau_{had}#tau_{had} Preselection'
    common_cuts = (
        PRESELECTION
        # & Cut(MET_CENTRALITY.format(pi / 4))
        )


class Category_Preselection_DEta_Control(Category_Preselection):
    is_control = True
    name = 'preselection_deta_control'


class Category_1J_Inclusive(Category_Preselection):
    name = '1j_inclusive'
    label = '#tau_{had}#tau_{had} Inclusive 1-Jet'
    common_cuts = Category_Preselection.common_cuts
    cuts = AT_LEAST_1JET
    norm_category = Category_Preselection
