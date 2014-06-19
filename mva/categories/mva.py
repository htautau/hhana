from math import pi
from rootpy.tree import Cut
from .common import (
    Category_Preselection,
    Category_Preselection_DEta_Control,
    CUTS_VBF, CUTS_BOOSTED, DETA_TAUS)
from .features import features_vbf, features_boosted


class Category_VBF_NO_DETAJJ_CUT(Category_Preselection):
    name = 'vbf'
    label = '#tau_{had}#tau_{had} VBF'
    common_cuts = Category_Preselection.common_cuts
    cuts = CUTS_VBF


class Category_VBF(Category_Preselection):
    name = 'vbf'
    label = '#tau_{had}#tau_{had} VBF'
    latex = '\\textbf{VBF}'
    color = 'red'
    linestyle = 'dotted'
    jk_number = 10
    common_cuts = Category_Preselection.common_cuts
    cuts = (
        CUTS_VBF
        & Cut('dEta_jets > 2.0')
        )
    features = features_vbf
    # train with only VBF mode
    signal_train_modes = ['VBF']
    norm_category = Category_Preselection


class Category_VBF_DEta_Control(Category_VBF):
    is_control = True
    name = 'vbf_deta_control'
    plot_label = 'Multijet CR'
    norm_category = Category_Preselection_DEta_Control
    #norm_category = Category_Preselection


class Category_Boosted(Category_Preselection):
    name = 'boosted'
    label = '#tau_{had}#tau_{had} Boosted'
    latex = '\\textbf{Boosted}'
    color = 'blue'
    linestyle = 'dashed'
    jk_number = 9
    common_cuts = Category_Preselection.common_cuts
    cuts = (
        (- Category_VBF.cuts)
        & CUTS_BOOSTED
        #& Cut(MET_CENTRALITY.format(pi / 6))
        )
    features = features_boosted
    # train with all modes (inherited from Category in base.py)
    #signal_train_modes =
    norm_category = Category_Preselection


class Category_Boosted_DEta_Control(Category_Boosted):
    is_control = True
    name = 'boosted_deta_control'
    label = '#tau_{had}#tau_{had} Boosted'
    plot_label = 'Multijet CR'
    #norm_category = Category_Preselection_DEta_Control
    norm_category = Category_Preselection


class Category_Rest(Category_Preselection):
    analysis_control = True
    name = 'rest'
    label = '#tau_{had}#tau_{had} Rest'
    latex = '\\textbf{Rest}'
    jk_number = 8
    common_cuts = Category_Preselection.common_cuts
    cuts = (- Category_Boosted.cuts) & (- Category_VBF.cuts) & DETA_TAUS
    norm_category = Category_Preselection
