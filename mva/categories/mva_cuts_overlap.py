# -- Create by Quentin Buat quentin(dot)buat(at)cern(dot)ch
# Implement an overlaping category between mva and cut-based analysis
#

from .mva import Category_Preselection
from .mva import Category_VBF
from .mva import Category_Boosted
from .cuts import Category_Cuts_VBF
from .cuts import Category_Cuts_Boosted
from .cuts import Category_Cuts_Preselection

class Category_Overlap_VBF(Category_Preselection):
    name ='mva_cut_based_overlap_vbf'
    label = '#tau_{had}#tau_{had} mva/cb overlap VBF'
    cuts = Category_VBF.cuts & Category_Cuts_VBF.cuts & Category_Cuts_VBF.common_cuts
    norm_category = Category_Preselection
    features = Category_VBF.features
    clf_category = Category_VBF
    limitbins = 40    
class Category_Overlap_Boosted(Category_Preselection):
    name ='mva_cut_based_overlap_boosted'
    label = '#tau_{had}#tau_{had} mva/cb overlap Boosted'
    cuts = Category_Boosted.cuts & Category_Cuts_Boosted.cuts & Category_Cuts_Boosted.common_cuts
    norm_category = Category_Preselection
    features = Category_Boosted.features
    clf_category = Category_Boosted
    limitbins = 40

class Category_Overlap_Preselection(Category_Cuts_Preselection):
    name='cut_based_mva_presel'
    label='#tau_{had}#tau_{had} cb&mva'
    cuts = (Category_Preselection.common_cuts) 
    norm_category = Category_Preselection

# ---> MVA VBF/Boosted and not preselected by CB
class Category_Cuts_VBF_NotMVA_Preselection(Category_Cuts_Preselection):
    name='cut_based_vbf_not_mva_presel'
    label='#tau_{had}#tau_{had} cb VBF&#overbar{mva presel}'
    cuts = (-Category_Preselection.common_cuts) & Category_Cuts_VBF.cuts
    norm_category = Category_Preselection
    features = Category_VBF.features

class Category_Cuts_Boosted_NotMVA_Preselection(Category_Cuts_Preselection):
    name='cut_based_boosted_notmva_presel'
    label='#tau_{had}#tau_{had} cb Boosted&#overbar{mva presel}'
    cuts = (-Category_Preselection.common_cuts) & Category_Cuts_Boosted.cuts
    norm_category = Category_Preselection
    features = Category_Boosted.features


# ---> MVA VBF and NotCuts (Presel/VBF/Boosted)
class Category_MVA_VBF_NotCuts_Preselection(Category_Preselection):
    name='mva_vbf_not_cut_presel'
    label='#tau_{had}#tau_{had} mva VBF&#overbar{cut presel}'
    cuts = (-Category_Cuts_Preselection.common_cuts) & Category_VBF.cuts
    norm_category = Category_Preselection
    features = Category_VBF.features

class Category_MVA_VBF_NotCuts_VBF(Category_Preselection):
    name='mva_vbf_not_cut_vbf'
    label='#tau_{had}#tau_{had} mva VBF&#overbar{cut vbf}'
    cuts = (-Category_Cuts_Preselection.common_cuts | -Category_Cuts_VBF.cuts) & Category_VBF.cuts
    norm_category = Category_Preselection
    features = Category_VBF.features

class Category_MVA_VBF_NotCuts_Boosted(Category_Preselection):
    name='mva_vbf_not_cut_boosted'
    label='#tau_{had}#tau_{had} mva VBF&#overbar{cut boosted}'
    cuts = (-Category_Cuts_Preselection.common_cuts | -Category_Cuts_Boosted.cuts) & Category_VBF.cuts
    norm_category = Category_Preselection
    features = Category_VBF.features

# ---> MVA Boosted and NotCuts (Presel/VBF/Boosted)
class Category_MVA_Boosted_NotCuts_Preselection(Category_Preselection):
    name='mva_boosted_not_cut_presel'
    label='#tau_{had}#tau_{had} mva Boosted&#overbar{cut presel}'
    cuts = (-Category_Cuts_Preselection.common_cuts) & Category_Boosted.cuts
    norm_category = Category_Preselection
    features = Category_Boosted.features

class Category_MVA_Boosted_NotCuts_VBF(Category_Preselection):
    name='mva_boosted_not_cut_vbf'
    label='#tau_{had}#tau_{had} mva Boosted&#overbar{cut vbf}'
    cuts = (-Category_Cuts_Preselection.common_cuts | -Category_Cuts_VBF.cuts) & Category_Boosted.cuts
    norm_category = Category_Preselection
    features = Category_VBF.features

class Category_MVA_Boosted_NotCuts_Boosted(Category_Preselection):
    name='mva_boosted_not_cut_boosted'
    label='#tau_{had}#tau_{had} mva Boosted&#overbar{cut boosted}'
    cuts = (-Category_Cuts_Preselection.common_cuts | -Category_Cuts_Boosted.cuts) & Category_Boosted.cuts
    norm_category = Category_Preselection
    features = Category_Boosted.features





