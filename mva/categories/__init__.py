from .mva import *
from .cuts import *
from .mva_cuts_overlap import *

CATEGORIES = {
    'cuts_presel': [
        Category_Cuts_Preselection,
        ],
    'cuts' : [
        Category_Cuts_VBF_LowDR,
        Category_Cuts_VBF_HighDR_Tight,
        Category_Cuts_VBF_HighDR_Loose,
        Category_Cuts_Boosted_Tight,
        Category_Cuts_Boosted_Loose,
#         Category_Cuts_Boosted,
#         Category_Cuts_VBF,
        ],
    'presel': [
        Category_Preselection,
        ],
    'presel_deta_controls': [
        Category_Preselection_DEta_Control,
        ],
    'mva': [
        Category_VBF,
        Category_Boosted,
    ],
    'mva_all': [
        Category_VBF,
        Category_Boosted,
        Category_Rest,
    ],
    'mva_deta_controls': [
        Category_VBF_DEta_Control,
        Category_Boosted_DEta_Control,
    ],
    'mva_workspace_controls': [
        Category_Rest,
    ],
    'overlap': [
    Category_Overlap_VBF,
    Category_Overlap_Boosted,
    ],
    'overlap_details': [
    Category_Cuts_Preselection,
    Category_Preselection,
    Category_Overlap_Preselection,
#     Category_Cuts_VBF_NotMVA_Preselection,
#     Category_Cuts_Boosted_NotMVA_Preselection,
    Category_VBF,
#     Category_MVA_VBF_NotCuts_Preselection,
#     Category_MVA_VBF_NotCuts_VBF,
#     Category_MVA_VBF_NotCuts_Boosted,
    Category_Cuts_VBF,
    Category_Overlap_VBF,
    Category_Boosted,
#     Category_MVA_Boosted_NotCuts_Preselection,
#     Category_MVA_Boosted_NotCuts_VBF,
#     Category_MVA_Boosted_NotCuts_Boosted,
    Category_Cuts_Boosted,
    Category_Overlap_Boosted,
    ]




}
