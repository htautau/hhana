# -- Create by Quentin Buat quentin(dot)buat(at)cern(dot)ch
# Implement an overlaping category between mva and cut-based analysis
#

from .mva import Category_Preselection
from .mva import Category_VBF
from .mva import Category_Boosted
from .cuts import Category_Cuts_VBF
from .cuts import Category_Cuts_Boosted


class Category_Overlap_VBF(Category_Preselection):
    name ='mva_cut_based_overlap_vbf'
    label = '#tau_{had}#tau_{had} mva/cb overlap VBF'
    cuts = Category_VBF.cuts & Category_Cuts_VBF.cuts
    norm_category = Category_Preselection
    features = Category_VBF.features
    clf_category = Category_VBF
    
class Category_Overlap_Boosted(Category_Preselection):
    name ='mva_cut_based_overlap_boosted'
    label = '#tau_{had}#tau_{had} mva/cb overlap Boosted'
    cuts = Category_Boosted.cuts & Category_Cuts_Boosted.cuts
    norm_category = Category_Preselection
    features = Category_Boosted.features
    clf_category = Category_Boosted
