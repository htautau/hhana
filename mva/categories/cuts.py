import math

from rootpy.tree import Cut

from .base import Category
from .selections import *
from .mva import Category_Preselection

# https://cds.cern.ch/record/1629891/files/ATL-COM-PHYS-2013-1558.pdf

# preselection cuts
COMMON_CUTS_CUTBASED = (
    LEAD_TAU_35 & SUBLEAD_TAU_25
    #& MET # <= no MET cut in cut-based preselection
    & Cut('%s > 0' % MMC_MASS)
    & Cut('0.6 < dR_tau1_tau2 < 2.8') # DIFFERENT THAN MVA (0.8 -> 0.6)
    & TAU_SAME_VERTEX
    & Cut('MET_bisecting || (dPhi_min_tau_MET < (0.2 * %f))' % math.pi)
    )

VBF_CUTS_CUTBASED = (
    MET & CUTS_2J
    & Cut('dEta_jets > 2.6')
    & Cut('mass_jet1_jet2 > 250000')
    & Cut('tau1_centrality > %f' % (1. / math.e))
    & Cut('tau2_centrality > %f' % (1. / math.e)))

BOOSTED_CUTS_CUTBASED = (
    MET
    & Cut('dR_tau1_tau2 < 2.5')
    & Cut('MET_bisecting || (dPhi_min_tau_MET < (0.1 * %f))' % math.pi)
    & Cut('resonance_pt > 80000'))

INF = 1E100


# Cut-based categories

class Category_Cuts_Preselection(Category):
    name = 'cuts_preselection'
    label = '#tau_{had}#tau_{had} Cut-based Preselection'
    common_cuts = COMMON_CUTS_CUTBASED
    norm_category = Category_Preselection


class Category_Cuts_VBF_LowDR(Category_Cuts_Preselection):
    name = 'cuts_vbf_lowdr'
    label = '#tau_{had}#tau_{had} Cut-based VBF Low #delta R'
    cuts = (VBF_CUTS_CUTBASED
        & Cut('dR_tau1_tau2 < 1.5') & Cut('resonance_pt > 140000'))
    limitbins = [0,64,80,92,104,116,132,176,INF]


class Category_Cuts_VBF_HighDR_Tight(Category_Cuts_Preselection):
    name = 'cuts_vbf_highdr_tight'
    label = '#tau_{had}#tau_{had} Cut-based VBF High #delta R Tight'
    cuts = (VBF_CUTS_CUTBASED
        & (Cut('dR_tau1_tau2 > 1.5') | Cut('resonance_pt < 140000'))
        & Cut('mass_jet1_jet2 > (-250000 * dEta_jets + 1550000)'))
    limitbins = [0,64,80,92,104,116,132,152,176,INF]


class Category_Cuts_VBF_HighDR_Loose(Category_Cuts_Preselection):
    name = 'cuts_vbf_highdr_loose'
    label = '#tau_{had}#tau_{had} Cut-based VBF High #delta R Loose'
    cuts = (VBF_CUTS_CUTBASED
        & (Cut('dR_tau1_tau2 > 1.5') | Cut('resonance_pt < 140000'))
        & Cut('mass_jet1_jet2 < (-250000 * dEta_jets + 1550000)'))
    limitbins = [0,64,80,92,104,116,132,152,176,INF]


class Category_Cuts_Boosted_Tight(Category_Cuts_Preselection):
    name = 'cuts_boosted_tight'
    label = '#tau_{had}#tau_{had} Cut-based Boosted Tight'
    cuts = ((- VBF_CUTS_CUTBASED) & BOOSTED_CUTS_CUTBASED
        & ((Cut('resonance_pt > (-200000 * dR_tau1_tau2 + 400000)') & Cut('resonance_pt > 140000')) | Cut('resonance_pt > 200000')))
    limitbins = [0,64,72,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,152,160,168,176,184,200,INF]


class Category_Cuts_Boosted_Loose(Category_Cuts_Preselection):
    name = 'cuts_boosted_loose'
    label = '#tau_{had}#tau_{had} Cut-based Boosted Loose'
    cuts = ((- VBF_CUTS_CUTBASED) & BOOSTED_CUTS_CUTBASED
        & Cut('resonance_pt > (-200000 * dR_tau1_tau2 + 400000)') & Cut('resonance_pt < 140000')
        & Cut('dEta_tau1_tau2 < 1'))
    limitbins = [0,72,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,152,176,184,INF]
