from rootpy.tree import Cut
from math import pi

from ..base import Category
from ... import MMC_MASS
# All basic cut definitions are here


TAU1_LOOSE = Cut('tau_0_jet_bdt_loose==1')
TAU1_MEDIUM = Cut('tau_0_jet_bdt_medium==1')
TAU1_TIGHT = Cut('tau_0_jet_bdt_tight==1')
TAU1_ANTI_MEDIUM = TAU1_LOOSE & -TAU1_MEDIUM

TAU2_LOOSE = Cut('tau_1_jet_bdt_loose==1')
TAU2_MEDIUM = Cut('tau_1_jet_bdt_medium==1')
TAU2_TIGHT = Cut('tau_1_jet_bdt_tight==1')
TAU2_ANTI_MEDIUM = TAU2_LOOSE & -TAU2_MEDIUM


ID_MEDIUM = TAU1_MEDIUM & TAU2_MEDIUM
ANTI_ID_MEDIUM = TAU1_ANTI_MEDIUM & TAU2_ANTI_MEDIUM

ID_TIGHT = TAU1_TIGHT & TAU2_TIGHT
ID_MEDIUM_TIGHT = (TAU1_MEDIUM & TAU2_TIGHT) | (TAU1_TIGHT & TAU2_MEDIUM)
# ID cuts for control region where both taus are medium but not tight
ID_MEDIUM_NOT_TIGHT = (TAU1_MEDIUM & -TAU1_TIGHT) & (TAU2_MEDIUM & -TAU2_TIGHT)

# 2015/08/25 DTemple: started trying to apply Quentin's suggestion from yesterday, putting attempt
# on hold ...
# TAU1_NOTMEDIUM = Cut('tau1_JetBDTSigMedium!=1')
# TAU2_NOTMEDIUM = Cut('tau2_JetBDTSigMedium!=1')
# OS_ID_MEDIUM_TIGHT
# OS_NOT_MEDIUM = TAU1_NOTMEDIUM & TAU2_NOTMEDIUM

TAU_SAME_VERTEX = Cut('tau_same_vertex')

LEAD_TAU_40 = Cut('tau_0_pt > 40')
SUBLEAD_TAU_30 = Cut('tau_1_pt > 30')
# 2015/08/25 DTemple: started trying to modify to match Quentin's version on GitHub ... too much
# different, don't understand well enough, putting attempt on hold ...
# LEAD_TAU_35    = Cut('tau0_pt > 35')
# SUBLEAD_TAU_25 = Cut('tau1_pt > 25')

LEAD_JET_50 = Cut('jet_0_pt > 50')
SUBLEAD_JET_30 = Cut('jet_1_pt > 30')
AT_LEAST_1JET = Cut('jet_0_pt > 30')

CUTS_2J = LEAD_JET_50 & SUBLEAD_JET_30
CUTS_1J = LEAD_JET_50 & (- SUBLEAD_JET_30)
CUTS_0J = (- LEAD_JET_50)

MET = Cut('met_et > 20')
DR_TAUS = Cut('0.8 < tau_tau_dr < 2.4')
DETA_TAUS = Cut('tau_tau_deta < 1.5')
DETA_TAUS_CR = Cut('dEta_tau1_tau2 > 1.5')
RESONANCE_PT = Cut('tau_tau_vect_sum_pt > 100')

# use .format() to set centality value
MET_CENTRALITY = 'tau_tau_met_bisect==1 || (tau_tau_met_min_dphi < {0})'

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
# 2015/08/25 DTemple: started trying to modify to match Quentin's version on GitHub ... too much
# different, don't understand well enough, putting attempt on hold ...
# PRESELECTION = (
#     LEAD_TAU_35 & SUBLEAD_TAU_25
#     & ID_MEDIUM_TIGHT
#     & MET
#     & Cut('%s > 0' % MMC_MASS)
#     & DR_TAUS
#     & TAU_SAME_VERTEX
#     )

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
