from math import pi
from rootpy.tree import Cut
from .common import (
    Category_Preselection_lh,
    CUTS_VBF, CUTS_BOOSTED)

class Category_VBF_lh(Category_Preselection_lh):
    name = 'vbf_lh'
    label = 'e#tau_{had} + #mu#tau_{had} VBF'
    latex = '\\textbf{VBF}'
    color = 'red'
    linestyle = 'dotted'
    common_cuts = Category_Preselection_lh.common_cuts
    cuts = (
        CUTS_VBF
        )

class Category_Boosted_lh(Category_Preselection_lh):
    name = 'boosted_lh'
    label = 'e#tau_{had} + #mu#tau_{had} Boosted'
    latex = '\\textbf{Boosted}'
    color = 'blue'
    linestyle = 'dashed'
    common_cuts = Category_Preselection_lh.common_cuts
    cuts = (
        (- Category_VBF_lh.cuts)
        & CUTS_BOOSTED
        )


