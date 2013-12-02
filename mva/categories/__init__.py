from .mva import *
from .cuts import *

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
    ]
}
