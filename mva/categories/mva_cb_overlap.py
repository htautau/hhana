# -- Create by Quentin Buat quentin(dot)buat(at)cern(dot)ch
# Implement an overlaping category between mva and cut-based analysis
#
from .common import Category_Preselection
from .mva import Category_VBF, Category_Boosted, Category_MVA
from .cb import Category_Cuts_VBF, Category_Cuts_Boosted, Category_Cuts

# ---> Cut_VBF and MVA (VBF,Boosted,Presel)
class Category_Cut_VBF_MVA_VBF(Category_Preselection):
    name ='mva_vbf_and_cut_vbf'
    label = '#tau_{had}#tau_{had} mva/cb overlap VBF'
    cuts = Category_VBF.cuts & Category_Cuts_VBF.cuts & Category_Cuts_VBF.common_cuts
    norm_category = Category_Preselection
    features = Category_VBF.features
    clf_category = Category_VBF
    limitbins = 40

class Category_Cut_VBF_MVA_Boosted(Category_Preselection):
    name ='mva_boosted_and_cut_vbf'
    label = '#tau_{had}#tau_{had} mva boosted cb VBF overlap'
    cuts = Category_Boosted.cuts & Category_Cuts_VBF.cuts & Category_Cuts_VBF.common_cuts
    norm_category = Category_Preselection
    features = Category_Boosted.features
    clf_category = Category_Boosted
    limitbins = 40

class Category_Cut_VBF_MVA_Presel(Category_Preselection):
    name ='mva_vbf_and_cut_presel'
    label = '#tau_{had}#tau_{had} mva VBF cb presel overlap'
    cuts = Category_Cuts_VBF.cuts & Category_Cuts_VBF.common_cuts
    norm_category = Category_Preselection

class Category_Cut_VBF_Not_MVA(Category_Preselection):
    name='cut_vbf_and_not_mva'
    cuts = Category_Cuts_VBF.cuts & Category_Cuts_VBF.common_cuts & -Category_MVA.cuts
    norm_category = Category_Preselection
    
# ---> Cut_Boosted and MVA (VBF,Boosted,Presel)
class Category_Cut_Boosted_MVA_VBF(Category_Preselection):
    name ='mva_vbf_and_cut_boosted'
    label = '#tau_{had}#tau_{had} mva boosted cb VBF overlap'
    cuts = Category_VBF.cuts & Category_Cuts_Boosted.cuts & Category_Cuts_Boosted.common_cuts
    norm_category = Category_Preselection
    features = Category_VBF.features
    clf_category = Category_VBF
    limitbins = 40

class Category_Cut_Boosted_MVA_Boosted(Category_Preselection):
    name ='mva_boosted_and_cut_boosted'
    label = '#tau_{had}#tau_{had} mva/cb overlap Boosted'
    cuts = Category_Boosted.cuts & Category_Cuts_Boosted.cuts & Category_Cuts_Boosted.common_cuts
    norm_category = Category_Preselection
    features = Category_Boosted.features
    clf_category = Category_Boosted
    limitbins = 40

class Category_Cut_Boosted_Not_MVA(Category_Preselection):
    name='cut_boosted_and_not_mva'
    cuts = Category_Cuts_Boosted.cuts & Category_Cuts_Boosted.common_cuts & -Category_MVA.cuts
    norm_category = Category_Preselection

class Category_Cut_Boosted_MVA_Presel(Category_Preselection):
    name ='mva_presel_and_cut_boosted'
    label = '#tau_{had}#tau_{had} mva/cb overlap Boosted'
    cuts = Category_Cuts_Boosted.cuts & Category_Cuts_Boosted.common_cuts
    norm_category = Category_Preselection


# ---> Cut_Presel and MVA (VBF,Boosted,Presel)
class Category_Cut_Presel_MVA_VBF(Category_Preselection):
    name  = 'mva_vbf_and_cut_presel'
    label = ''
    cuts  = Category_VBF.common_cuts & Category_VBF.cuts
    norm_category = Category_Preselection


class Category_Cut_Presel_MVA_Boosted(Category_Preselection):
    name  = 'mva_boosted_and_cut_presel'
    label = ''
    cuts  = Category_Boosted.common_cuts & Category_Boosted.cuts
    norm_category = Category_Preselection


class Category_Cut_Presel_MVA_Presel(Category_Preselection):
    name  = 'cut_presel_and_mva_presel'
    label ='#tau_{had}#tau_{had} cb&mva'
    cuts  = Category_Preselection.common_cuts
    norm_category = Category_Preselection


# ---> Cut_Presel and NOT MVA (VBF,Boosted,Presel)
class Category_Cut_Presel_Not_MVA_VBF(Category_Preselection):
    name  = 'not_mva_vbf_and_cut_presel'
    label = ''
    cuts  = - (Category_VBF.common_cuts & Category_VBF.cuts)
    norm_category = Category_Preselection


class Category_Cut_Presel_Not_MVA_Boosted(Category_Preselection):
    name  = 'not_mva_boosted_and_cut_presel'
    label = ''
    cuts  = - (Category_Boosted.common_cuts & Category_Boosted.cuts)
    norm_category = Category_Preselection


class Category_Cut_Presel_Not_MVA_Presel(Category_Preselection):
    name  = 'not_mva_presel_and_cut_presel'
    label = ''
    cuts  = - (Category_Preselection.common_cuts)


# ---> Cut_VBF and NOT MVA (VBF,Boosted,Presel)
class Category_Cut_VBF_Not_MVA_VBF(Category_Preselection):
    name  = 'not_mva_vbf_and_cut_vbf'
    label = ''
    cuts  = - (Category_VBF.common_cuts & Category_VBF.cuts) & Category_Cuts_VBF.cuts
    norm_category = Category_Preselection


class Category_Cut_VBF_Not_MVA_Boosted(Category_Preselection):
    name  = 'not_mva_boosted_and_cut_vbf'
    label = ''
    cuts  = - (Category_Boosted.common_cuts & Category_Boosted.cuts) & Category_Cuts_VBF.cuts
    norm_category = Category_Preselection


class Category_Cut_VBF_Not_MVA_Presel(Category_Preselection):
    name  = 'not_mva_presel_and_cut_vbf'
    label = ''
    cuts  = - (Category_Preselection.common_cuts) & Category_Cuts_VBF.cuts


# ---> Cut_Boosted and NOT MVA (VBF,Boosted,Presel)
class Category_Cut_Boosted_Not_MVA_VBF(Category_Preselection):
    name  = 'not_mva_vbf_and_cut_boosted'
    label = ''
    cuts  = - (Category_VBF.common_cuts & Category_VBF.cuts) & Category_Cuts_Boosted.cuts
    norm_category = Category_Preselection


class Category_Cut_Boosted_Not_MVA_Boosted(Category_Preselection):
    name  = 'not_mva_boosted_and_cut_boosted'
    label = ''
    cuts  = - (Category_Boosted.common_cuts & Category_Boosted.cuts) & Category_Cuts_Boosted.cuts
    norm_category = Category_Preselection


class Category_Cut_Boosted_Not_MVA_Presel(Category_Preselection):
    name  = 'not_mva_presel_and_cut_boosted'
    label = ''
    cuts  = - (Category_Preselection.common_cuts) & Category_Cuts_Boosted.cuts


# ---> MVA_Presel and NOT Cut (VBF,Boosted,Presel)
class Category_MVA_Presel_Not_Cut_VBF    (Category_Preselection):
    name  = 'mva_presel_and_not_cut_vbf'
    label = ''
    cuts  = - (Category_Cuts_VBF.common_cuts & Category_Cuts_VBF.cuts)
    norm_category = Category_Preselection


class Category_MVA_Presel_Not_Cut_Boosted(Category_Preselection):
    name  = 'mva_presel_and_not_cut_boosted'
    label = ''
    cuts  = - (Category_Cuts_Boosted.common_cuts & Category_Cuts_Boosted.cuts)
    norm_category = Category_Preselection


class Category_MVA_Presel_Not_Cut_Presel (Category_Preselection):
    name  = 'mva_presel_and_not_cut_presel'
    label = ''
    cuts  = - (Category_Preselection.common_cuts)
    norm_category = Category_Preselection



# ---> MVA_VBF and NOT Cut (VBF,Boosted,Presel)
class Category_MVA_VBF_Not_Cut_VBF    (Category_Preselection):
    name  = 'mva_vbf_and_not_cut_vbf'
    label = ''
    cuts  = - (Category_Cuts_VBF.common_cuts & Category_Cuts_VBF.cuts) & Category_VBF.cuts
    norm_category = Category_Preselection
    clf_category = Category_VBF
class Category_MVA_VBF_Not_Cut_Boosted(Category_Preselection):
    name  = 'mva_vbf_and_not_cut_boosted'
    label = ''
    cuts  = - (Category_Cuts_Boosted.common_cuts & Category_Cuts_Boosted.cuts)  & Category_VBF.cuts
    norm_category = Category_Preselection
    clf_category = Category_VBF

class Category_MVA_VBF_Not_Cut_Presel (Category_Preselection):
    name  = 'mva_vbf_and_not_cut_presel'
    label = ''
    cuts  = - (Category_Preselection.common_cuts) & Category_VBF.cuts
    norm_category = Category_Preselection

class Category_MVA_VBF_Not_Cut(Category_Preselection):
    name='mva_vbf_and_not_cut'
    cuts = Category_VBF.cuts & Category_VBF.common_cuts & -Category_Cuts.cuts
    norm_category = Category_Preselection
    clf_category = Category_VBF


# ---> MVA_Boosted and NOT Cut (VBF,Boosted,Presel)
class Category_MVA_Boosted_Not_Cut_VBF    (Category_Preselection):
    name  = 'mva_boosted_and_not_cut_vbf'
    label = ''
    cuts  = - (Category_Cuts_VBF.common_cuts & Category_Cuts_VBF.cuts) & Category_Boosted.cuts
    norm_category = Category_Preselection
class Category_MVA_Boosted_Not_Cut_Boosted(Category_Preselection):
    name  = 'mva_boosted_and_not_cut_boosted'
    label = ''
    cuts  = - (Category_Cuts_Boosted.common_cuts & Category_Cuts_Boosted.cuts)  & Category_Boosted.cuts
    norm_category = Category_Preselection
class Category_MVA_Boosted_Not_Cut_Presel (Category_Preselection):
    name  = 'mva_boosted_and_not_cut_presel'
    label = ''
    cuts  = - (Category_Preselection.common_cuts) & Category_Boosted.cuts
    norm_category = Category_Preselection

class Category_MVA_Boosted_Not_Cut(Category_Preselection):
    name='mva_boosted_and_not_cut'
    cuts = Category_Boosted.cuts & Category_Boosted.common_cuts & -Category_Cuts.cuts
    norm_category = Category_Preselection
    clf_category = Category_Boosted

# # ---> MVA VBF/Boosted and not preselected by CB
# class Category_Cuts_VBF_NotMVA_Preselection(Category_Preselection):
#     name='cut_based_vbf_not_mva_presel'
#     label='#tau_{had}#tau_{had} cb VBF&#overbar{mva presel}'
#     cuts = (-Category_Preselection.common_cuts) & Category_Cuts_VBF.cuts
#     norm_category = Category_Preselection
#     features = Category_VBF.features

# class Category_Cuts_Boosted_NotMVA_Preselection(Category_Preselection):
#     name='cut_based_boosted_notmva_presel'
#     label='#tau_{had}#tau_{had} cb Boosted&#overbar{mva presel}'
#     cuts = (-Category_Preselection.common_cuts) & Category_Cuts_Boosted.cuts
#     norm_category = Category_Preselection
#     features = Category_Boosted.features


# # ---> MVA VBF and NotCuts (Presel/VBF/Boosted)
# class Category_MVA_VBF_NotCuts_Preselection(Category_Preselection):
#     name='mva_vbf_not_cut_presel'
#     label='#tau_{had}#tau_{had} mva VBF&#overbar{cut presel}'
#     cuts = (-Category_Preselection.common_cuts) & Category_VBF.cuts
#     norm_category = Category_Preselection
#     features = Category_VBF.features

# class Category_MVA_VBF_NotCuts_VBF(Category_Preselection):
#     name='mva_vbf_not_cut_vbf'
#     label='#tau_{had}#tau_{had} mva VBF&#overbar{cut vbf}'
#     cuts = (-Category_Preselection.common_cuts | -Category_Cuts_VBF.cuts) & Category_VBF.cuts
#     norm_category = Category_Preselection
#     features = Category_VBF.features

# class Category_MVA_VBF_NotCuts_Boosted(Category_Preselection):
#     name='mva_vbf_not_cut_boosted'
#     label='#tau_{had}#tau_{had} mva VBF&#overbar{cut boosted}'
#     cuts = (-Category_Preselection.common_cuts | -Category_Cuts_Boosted.cuts) & Category_VBF.cuts
#     norm_category = Category_Preselection
#     features = Category_VBF.features

# # ---> MVA Boosted and NotCuts (Presel/VBF/Boosted)
# class Category_MVA_Boosted_NotCuts_Preselection(Category_Preselection):
#     name='mva_boosted_not_cut_presel'
#     label='#tau_{had}#tau_{had} mva Boosted&#overbar{cut presel}'
#     cuts = (-Category_Preselection.common_cuts) & Category_Boosted.cuts
#     norm_category = Category_Preselection
#     features = Category_Boosted.features

# class Category_MVA_Boosted_NotCuts_VBF(Category_Preselection):
#     name='mva_boosted_not_cut_vbf'
#     label='#tau_{had}#tau_{had} mva Boosted&#overbar{cut vbf}'
#     cuts = (-Category_Preselection.common_cuts | -Category_Cuts_VBF.cuts) & Category_Boosted.cuts
#     norm_category = Category_Preselection
#     features = Category_VBF.features

# class Category_MVA_Boosted_NotCuts_Boosted(Category_Preselection):
#     name='mva_boosted_not_cut_boosted'
#     label='#tau_{had}#tau_{had} mva Boosted&#overbar{cut boosted}'
#     cuts = (-Category_Preselection.common_cuts | -Category_Cuts_Boosted.cuts) & Category_Boosted.cuts
#     norm_category = Category_Preselection
#     features = Category_Boosted.features
