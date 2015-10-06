from math import pi
from rootpy.tree import Cut
from .common import (
    Category_Preselection_lh, Category_Preselection_lh_NoBveto,
    CUTS_VBF, CUTS_BOOSTED, CUTS_WPLUSJETS_CR, CUTS_Ztt_CR, CUTS_top_CR)

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

class Category_wplusjets_CR_lh(Category_Preselection_lh):
    name = 'wplusjets_CR_lh'
    label = 'e#tau_{had} + #mu#tau_{had} '
    latex = '\\textbf{wplusjets_CR}'
    color = 'red'
    linestyle = 'dotted'
    common_cuts = Category_Preselection_lh.common_cuts
    cuts = (
        CUTS_WPLUSJETS_CR
        )

class Category_Ztautau_CR_lh(Category_Preselection_lh):
    name = 'Ztautau_CR_lh'
    label = 'e#tau_{had} + #mu#tau_{had} '
    latex = '\\textbf{Ztautau_CR}'
    color = 'red'
    linestyle = 'dotted'
    common_cuts = Category_Preselection_lh.common_cuts
    cuts = (
        CUTS_Ztt_CR
        )

class Category_Top_CR_lh(Category_Preselection_lh_NoBveto):
    name = 'Top_CR_lh'
    label = 'e#tau_{had} + #mu#tau_{had} '
    latex = '\\textbf{Top_CR}'
    color = 'red'
    linestyle = 'dotted'
    common_cuts = Category_Preselection_lh_NoBveto.common_cuts
    cuts = (
        CUTS_top_CR
        )

class Category_VBF_wplusjets_CR_lh(Category_Preselection_lh):
    name = 'VBF_wplusjets_CR_lh'
    label = 'e#tau_{had} + #mu#tau_{had} '
    latex = '\\textbf{VBF_wplusjets_CR}'
    color = 'red'
    linestyle = 'dotted'
    common_cuts = Category_Preselection_lh.common_cuts
    cuts = (
        CUTS_WPLUSJETS_CR
        & CUTS_VBF
        )

class Category_VBF_Ztautau_CR_lh(Category_Preselection_lh):
    name = 'VBF_Ztautau_CR_lh'
    label = 'e#tau_{had} + #mu#tau_{had} '
    latex = '\\textbf{VBF_Ztautau_CR}'
    color = 'red'
    linestyle = 'dotted'
    common_cuts = Category_Preselection_lh.common_cuts
    cuts = (
        CUTS_Ztt_CR
        & CUTS_VBF
        )

class Category_VBF_Top_CR_lh(Category_Preselection_lh_NoBveto):
    name = 'VBF_Top_CR_lh'
    label = 'e#tau_{had} + #mu#tau_{had} '
    latex = '\\textbf{VBF_Top_CR}'
    color = 'red'
    linestyle = 'dotted'
    common_cuts = Category_Preselection_lh_NoBveto.common_cuts
    cuts = (
        CUTS_top_CR
        & CUTS_VBF
        )

class Category_Boosted_wplusjets_CR_lh(Category_Preselection_lh):
    name = 'Boosted_wplusjets_CR_lh'
    label = 'e#tau_{had} + #mu#tau_{had} '
    latex = '\\textbf{Boosted_wplusjets_CR}'
    color = 'red'
    linestyle = 'dotted'
    common_cuts = Category_Preselection_lh.common_cuts
    cuts = (
        CUTS_WPLUSJETS_CR
        & CUTS_BOOSTED
        )


class Category_Boosted_Ztautau_CR_lh(Category_Preselection_lh):
    name = 'Boosted_Ztautau_CR_lh'
    label = 'e#tau_{had} + #mu#tau_{had} '
    latex = '\\textbf{Boosted_Ztautau_CR}'
    color = 'red'
    linestyle = 'dotted'
    common_cuts = Category_Preselection_lh.common_cuts
    cuts = (
        CUTS_Ztt_CR
        & CUTS_BOOSTED
        )

class Category_Boosted_Top_CR_lh(Category_Preselection_lh_NoBveto):
    name = 'Boosted_Top_CR_lh'
    label = 'e#tau_{had} + #mu#tau_{had} '
    latex = '\\textbf{Boosted_Top_CR}'
    color = 'red'
    linestyle = 'dotted'
    common_cuts = Category_Preselection_lh_NoBveto.common_cuts
    cuts = (
        CUTS_top_CR
        & CUTS_BOOSTED
        )
