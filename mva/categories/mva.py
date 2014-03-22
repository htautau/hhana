from rootpy.tree import Cut
from .common import Category_Preselection, Category_Preselection_DEta_Control
from .selections import CUTS_VBF, CUTS_BOOSTED
from .features import features_2j, features_boosted, features_0j


class Category_VBF(Category_Preselection):
    name = 'vbf'
    label = '#tau_{had}#tau_{had} VBF'
    common_cuts = Category_Preselection.common_cuts
    cuts = (
        CUTS_VBF
        & Cut('dEta_jets > 2.0')
        #& Cut('resonance_pt > 40000')
        )
    #limitbins = 98
    limitbins = 40
    features = features_2j
    # train with only VBF
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
    common_cuts = Category_Preselection.common_cuts
    cuts = (
        (- Category_VBF.cuts)
        & CUTS_BOOSTED
        )
    #limitbins = 86
    limitbins = 40
    # warning: some variables will be undefined for some events
    features = features_boosted
    # train with all modes
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
    common_cuts = Category_Preselection.common_cuts
    cuts = (- Category_Boosted.cuts) & (- Category_VBF.cuts)
    limitbins = 10
    features = features_0j
    # train with all modes
    norm_category = Category_Preselection
    #workspace_min_clf = 0.
