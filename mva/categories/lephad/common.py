from rootpy.tree import Cut
from math import pi

from ..base import Category
# All basic cut definitions are here


IS_OPPOSITE_SIGN = Cut('is_opposite_sign==1')
IS_VBF = Cut('is_vbf_mva==1')
IS_BOOSTED = Cut('is_boosted_mva==1')

MET = Cut('met_et > 0')
# common preselection cuts
PRESELECTION = (
    IS_OPPOSITE_SIGN
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


class Category_Preselection_lh(Category):
    name = 'preselection_lh'
    label = 'e#tau_{had} + #mu#tau_{had} Preselection'
    common_cuts = (
        PRESELECTION
        )


