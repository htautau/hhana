from .base import Category
from .selections import *
from .features import *

# preselection cuts
COMMON_CUTS_MVA = (
    LEAD_TAU_35 & SUBLEAD_TAU_25
    & MET
    & Cut('%s > 0' % MMC_MASS)
    & Cut('0.8 < dR_tau1_tau2 < 2.6')
    & TAU_SAME_VERTEX
    # looser MET centrality
    & Cut('MET_bisecting || (dPhi_min_tau_MET < %f)' % (0.2 * math.pi))
    )

# additional cuts after preselection
CATEGORY_CUTS_MVA = Cut()

# MVA preselection categories

class Category_Preselection(Category):
    name = 'preselection'
    label = '#tau_{had}#tau_{had} Preselection'
    common_cuts = COMMON_CUTS_MVA


class Category_Preselection_DEta_Control(Category_Preselection):
    is_control = True
    name = 'preselection_deta_control'


class Category_VBF(Category_Preselection):
    name = 'vbf'
    label = '#tau_{had}#tau_{had} VBF'
    common_cuts = Category_Preselection.common_cuts & CATEGORY_CUTS_MVA
    cuts = (
        CUTS_2J
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
    common_cuts = Category_Preselection.common_cuts & CATEGORY_CUTS_MVA
    cuts = (
        (- Category_VBF.cuts)
        & Cut('resonance_pt > 80000')
        & Cut('MET_bisecting || (dPhi_min_tau_MET < %f)' % (0.1 * math.pi))
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
    common_cuts = Category_Preselection.common_cuts & CATEGORY_CUTS_MVA
    cuts = (- Category_Boosted.cuts) & (- Category_VBF.cuts)
    limitbins = 10
    features = features_0j
    # train with all modes
    norm_category = Category_Preselection
    #workspace_min_clf = 0.


class Category_1J_Inclusive(Category_Preselection):
    name = '1j_inclusive'
    label = '#tau_{had}#tau_{had} Inclusive 1-Jet'
    common_cuts = Category_Preselection.common_cuts & CATEGORY_CUTS_MVA
    cuts = AT_LEAST_1JET
    norm_category = Category_Preselection
