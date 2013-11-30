from .base import Category
from .selections import *

# preselection cuts
COMMON_CUTS_CUTBASED = (
    LEAD_TAU_35 & SUBLEAD_TAU_25
    #& MET <= no MET cut in cut-based preselection
    & Cut('%s > 0' % MMC_MASS)
    & Cut('0.6 < dR_tau1_tau2 < 2.8') # DIFFERENT THAN MVA (0.8 -> 0.6)
    & TAU_SAME_VERTEX
    & Cut('MET_bisecting || (dPhi_min_tau_MET < (0.2 * %f))' % math.pi)
    )

# Cut-based categories

class Category_Cuts_Preselection(Category):
    name = 'cut_preselection'
    label = '#tau_{had}#tau_{had} Cut-based Preselection'
    common_cuts = COMMON_CUTS_CUTBASED


class Category_Cuts_Preselection_DEta_Control(Category_Cuts_Preselection):
    name = 'cut_preselection_deta_control'
