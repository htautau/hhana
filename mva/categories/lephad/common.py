from rootpy.tree import Cut
from math import pi

from ..base import Category
# All basic cut definitions are here
TRIGGER = Cut('HLT_e24_lhmedium_iloose_L1EM18VH==1') | Cut('HLT_e24_lhmedium_iloose_L1EM20VH==1') | Cut('HLT_e24_lhmedium_nod0_iloose_L1EM18VH==1') | Cut('HLT_e24_lhmedium_nod0_iloose_L1EM20VH==1') | Cut('HLT_e24_medium_iloose_L1EM18VH==1') | Cut('HLT_e24_medium_iloose_L1EM20VH==1') | Cut('HLT_mu20_iloose_L1MU15==1') | Cut('HLT_mu24_iloose_L1MU15==1')
IS_OPPOSITE_SIGN = Cut('is_opposite_sign==1')
IS_VBF = Cut('is_vbf_mva==1')
IS_BOOSTED = Cut('is_boosted_mva==1')
BVETO = Cut('is_btagged == 0')
MET = Cut('met_reco_et > 0')
MT_LEP_MET = Cut('lephad_mt_lep0_met < 70000.')
MT_LEP_MET_WPLUSJETS_CR =Cut('lephad_mt_lep0_met > 70000')
MT_LEP_MET_TOP_CR =Cut('lephad_mt_lep0_met > 70000')
MT_LEP_MET_ZTT_CR =Cut('lephad_mt_lep0_met < 40000')
DITAU_MASS_ZTT_CR = Cut('lephad_mmc_mlm_m < 110000')

# common preselection cuts
PRESELECTION = (
    TRIGGER
    & IS_OPPOSITE_SIGN
    & MET 
    & BVETO
    & MT_LEP_MET
    # & LEPTON_IS_ELE
    )

# common preselection cuts without BVETO and without Mt cuts (useful for CR)
PRESELECTION_CR = (
    TRIGGER
    & IS_OPPOSITE_SIGN
    & MET 
    
    )

# VBF category cuts
CUTS_VBF = (
    IS_VBF
    )

# Boosted category cuts
CUTS_BOOSTED = (
    IS_BOOSTED
    )

# VBF_Wplusjets_CR
CUTS_WPLUSJETS_CR = (
    MT_LEP_MET_WPLUSJETS_CR
    )


# VBF_Ztautau_CR
CUTS_Ztt_CR = (
    MT_LEP_MET_ZTT_CR
    & DITAU_MASS_ZTT_CR
    )

# VBF_top_CR
CUTS_top_CR = (
    -BVETO
    & MT_LEP_MET_TOP_CR
    )



class Category_Preselection_lh(Category):
    name = 'preselection_lh'
    label = 'e#tau_{had} + #mu#tau_{had} Preselection'
    common_cuts = (
        PRESELECTION
        )

class Category_Preselection_lh_CR(Category):
    name = 'preselection_lh'
    label = 'e#tau_{had} + #mu#tau_{had} Preselection'
    common_cuts = (
        PRESELECTION_CR
        )

