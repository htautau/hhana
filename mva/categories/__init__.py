from .mva_categories import *
from .cuts_categories import *

CATEGORIES = {
    'cuts_presel': [
        Category_Cuts_Preselection,
        ],
    'cuts_presel_deta_controls': [
        Category_Cuts_Preselection_DEta_Control,
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
