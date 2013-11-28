from rootpy.tree import Cut
import math
from . import MMC_MASS

# All basic cut definitions

TAU1_MEDIUM = Cut('tau1_JetBDTSigMedium==1')
TAU2_MEDIUM = Cut('tau2_JetBDTSigMedium==1')
TAU1_TIGHT = Cut('tau1_JetBDTSigTight==1')
TAU2_TIGHT = Cut('tau2_JetBDTSigTight==1')

ID_MEDIUM = TAU1_MEDIUM & TAU2_MEDIUM
ID_TIGHT = TAU1_TIGHT & TAU2_TIGHT
ID_MEDIUM_TIGHT = (TAU1_MEDIUM & TAU2_TIGHT) | (TAU1_TIGHT & TAU2_MEDIUM)
# ID cuts for control region where both taus are medium but not tight
ID_MEDIUM_NOT_TIGHT = (TAU1_MEDIUM & -TAU1_TIGHT) & (TAU2_MEDIUM & -TAU2_TIGHT)

TAU_SAME_VERTEX = Cut('tau_same_vertex')

LEAD_TAU_35 = Cut('tau1_pt > 35000')
SUBLEAD_TAU_25 = Cut('tau2_pt > 25000')

LEAD_JET_50 = Cut('jet1_pt > 50000')
SUBLEAD_JET_30 = Cut('jet2_pt > 30000')
AT_LEAST_1JET = Cut('jet1_pt > 30000')

CUTS_2J = LEAD_JET_50 & SUBLEAD_JET_30
CUTS_1J = LEAD_JET_50 & (- SUBLEAD_JET_30)
CUTS_0J = (- LEAD_JET_50)

CUTS_VBF = Cut('dEta_jets > 2.0')
CUTS_BOOSTED = Cut('resonance_pt > 100000') # MeV

BAD_MASS = 75
MET = Cut('MET_et > 20000')

COMMON_CUTS_CUTBASED = (
    LEAD_TAU_35 & SUBLEAD_TAU_25
    #& MET <= no MET cut in cut-based preselection
    & Cut('%s > 0' % MMC_MASS)
    & Cut('0.8 < dR_tau1_tau2 < 2.8')
    & TAU_SAME_VERTEX
    & Cut('MET_bisecting || (dPhi_min_tau_MET < (0.2 * %f))' % math.pi)
    )

# preselection cuts
COMMON_CUTS_MVA = (
    LEAD_TAU_35 & SUBLEAD_TAU_25
    & MET
    & Cut('%s > 0' % MMC_MASS)
    & Cut('0.8 < dR_tau1_tau2 < 2.8')
    & TAU_SAME_VERTEX
    # looser MET centrality
    & Cut('MET_bisecting || (dPhi_min_tau_MET < %f)' % (math.pi / 2))
    )

# additional cuts after preselection
#CATEGORY_CUTS_MVA = (
#    Cut('%s > 80' % MMC_MASS)
#    )
CATEGORY_CUTS_MVA = Cut()

# TODO: possible new variable: ratio of core tracks to recounted tracks
# TODO: add new pi0 info (new variables?)

features_2j = [
    MMC_MASS,
    # !!! mass ditau + leading jet?
    'dEta_jets',
    #'dEta_jets_boosted', #
    'eta_product_jets',
    #'eta_product_jets_boosted', #
    'mass_jet1_jet2',
    #'sphericity', #
    #'aplanarity', #
    'tau1_centrality',
    'tau2_centrality',
    #'cos_theta_tau1_tau2', #
    'dR_tau1_tau2',
    #'tau1_BDTJetScore',
    #'tau2_BDTJetScore',
    #'tau1_x', #
    #'tau2_x', #
    'MET_centrality',
    'vector_sum_pt',
    #'sum_pt_full', #
    #'resonance_pt',
    # !!! eta centrality of 3rd jet
]

features_boosted = [
    MMC_MASS,
    # !!! mass ditau + leading jet?
    #'dEta_jets',
    #'dEta_jets_boosted', #
    #'eta_product_jets',
    #'eta_product_jets_boosted', #
    #'mass_jet1_jet2',
    #'sphericity', #
    #'aplanarity', #
    #'tau1_centrality',
    #'tau2_centrality',
    #'tau1_centrality_boosted', #
    #'tau2_centrality_boosted', #
    #'cos_theta_tau1_tau2', #
    'dR_tau1_tau2',
    #'tau1_BDTJetScore',
    #'tau2_BDTJetScore',
    'tau1_collinear_momentum_fraction', #  <= ADD BACK IN
    'tau2_collinear_momentum_fraction', #  <= ADD BACK IN
    'MET_centrality',
    #'resonance_pt',
    'sum_pt_full',
    'tau_pt_ratio',
    # !!! eta centrality of 3rd jet
]

features_1j = [
    MMC_MASS,
    # !!! mass ditau + leading jet?
    #'sphericity',
    #'aplanarity',
    #'cos_theta_tau1_tau2',
    'dR_tau1_tau2',
    #'tau1_BDTJetScore',
    #'tau2_BDTJetScore',
    'tau1_collinear_momentum_fraction',
    'tau2_collinear_momentum_fraction',
    'MET_centrality',
    'sum_pt_full',
    'tau_pt_ratio',
    #'resonance_pt',
]

features_0j = [
    MMC_MASS,
    #'cos_theta_tau1_tau2',
    'dR_tau1_tau2',
    #'tau1_BDTJetScore',
    #'tau2_BDTJetScore',
    'tau1_collinear_momentum_fraction',
    'tau2_collinear_momentum_fraction',
    'MET_centrality',
    'sum_pt_full',
    'tau_pt_ratio',
    #'resonance_pt',
]


class CategoryMeta(type):
    """
    Metaclass for all categories
    """
    CATEGORY_REGISTRY = {}
    def __new__(cls, name, bases, dct):

        if name in CategoryMeta.CATEGORY_REGISTRY:
            raise ValueError("Multiple categories with the same name: %s" % name)
        cat = type.__new__(cls, name, bases, dct)
        # register the category
        CategoryMeta.CATEGORY_REGISTRY[name] = cat
        return cat


class Category(object):

    __metaclass__ = CategoryMeta

    # common attrs for all categories. Override in subclasses
    analysis_control = False
    is_control = False
    # category used for normalization
    norm_category = None
    qcd_shape_region = 'nOS' # no track cut
    target_region = 'OS_TRK'
    cuts = Cut()
    common_cuts = Cut()
    from . import samples
    train_signal_modes = samples.Higgs.MODES[:]
    clf_bins = 8
    # only unblind up to this number of bins in half-blind mode
    # flat, onebkg or constant (see mva/stats/utils.py)
    limitbinning = 'constant'
    plot_label = None

    @classmethod
    def get_cuts(cls, year, deta_cut=True):
        cuts = cls.cuts & cls.common_cuts
        if hasattr(cls, 'year_cuts') and year in cls.year_cuts:
            cuts &= cls.year_cuts[year]
        if 'DEta_Control' in cls.__name__:
            cuts &= Cut('dEta_tau1_tau2 >= 1.5')
        elif deta_cut:
            cuts &= Cut('dEta_tau1_tau2 < 1.5')
        # TODO
        #if 'ID_Control' in cls.__name__ or 'FF' in region:
        #    cuts &= TAUS_FAIL
        #else:
        #    cuts &= TAUS_PASS
        return cuts

    @classmethod
    def get_parent(cls):
        if cls.is_control:
            return cls.__bases__[0]
        return cls


# Cut-based categories

class Category_Cuts_Preselection(Category):
    name = 'cut_preselection'
    label = r'$\tau_{had}\tau_{had}$ Cut-based Preselection'
    root_label = '#tau_{had}#tau_{had} Cut-based Preselection'
    common_cuts = COMMON_CUTS_CUTBASED


class Category_Cuts_Preselection_DEta_Control(Category_Cuts_Preselection):
    name = 'cut_preselection_deta_control'
    label = r'$\tau_{had}\tau_{had}$ $\Delta \eta_{\tau_{1},\/\tau_{2}} \geq 1.5$ Control Region at Cut-based Preselection'


# MVA preselection categories

class Category_Preselection(Category):
    name = 'preselection'
    label = r'$\tau_{had}\tau_{had}$ Preselection'
    root_label = '#tau_{had}#tau_{had} Preselection'
    common_cuts = COMMON_CUTS_MVA
    #cuts = Cut('theta_tau1_tau2 > 0.6')


class Category_Preselection_DEta_Control(Category_Preselection):
    is_control = True
    name = 'preselection_deta_control'
    label = r'$\tau_{had}\tau_{had}$ $\Delta \eta_{\tau_{1},\/\tau_{2}} \geq 1.5$ Control Region at Preselection'


class Category_VBF(Category_Preselection):
    name = 'vbf'
    label = r'$\tau_{had}\tau_{had}$ VBF'
    root_label = '#tau_{had}#tau_{had} VBF'
    common_cuts = Category_Preselection.common_cuts & CATEGORY_CUTS_MVA
    cuts = CUTS_VBF & CUTS_2J & Cut('resonance_pt > 40000')
    fitbins = 5
    #limitbins = 98
    limitbins = 40
    features = features_2j
    # train with only VBF
    signal_train_modes = ['VBF']
    norm_category = Category_Preselection


class Category_VBF_DEta_Control(Category_VBF):
    is_control = True
    name = 'vbf_deta_control'
    label = r'$\tau_{had}\tau_{had}$ VBF Category $\Delta \eta_{\tau_{1},\/\tau_{2}} \geq 1.5$ Control Region'
    plot_label = 'Multijet CR'
    norm_category = Category_Preselection_DEta_Control
    #norm_category = Category_Preselection


class Category_Boosted(Category_Preselection):
    name = 'boosted'
    label = r'$\tau_{had}\tau_{had}$ Boosted'
    root_label = '#tau_{had}#tau_{had} Boosted'
    common_cuts = Category_Preselection.common_cuts & CATEGORY_CUTS_MVA
    cuts = CUTS_BOOSTED & (- Category_VBF.cuts)
    fitbins = 5
    #limitbins = 86
    limitbins = 40
    # warning: some variables will be undefined for some events
    features = features_boosted
    # train with all modes
    norm_category = Category_Preselection


class Category_Boosted_DEta_Control(Category_Boosted):
    is_control = True
    name = 'boosted_deta_control'
    label = r'$\tau_{had}\tau_{had}$ Boosted Category $\Delta \eta_{\tau_{1},\/\tau_{2}} \geq 1.5$ Control Region'
    root_label = '#tau_{had}#tau_{had} Boosted'
    plot_label = 'Multijet CR'
    #norm_category = Category_Preselection_DEta_Control
    norm_category = Category_Preselection


class Category_Rest(Category_Preselection):
    analysis_control = True
    name = 'rest'
    label = r'$\tau_{had}\tau_{had}$ Rest'
    root_label = '#tau_{had}#tau_{had} Rest'
    common_cuts = Category_Preselection.common_cuts & CATEGORY_CUTS_MVA
    cuts = (- Category_Boosted.cuts) & (- Category_VBF.cuts)
    fitbins = 8
    limitbins = 10
    features = features_0j
    # train with all modes
    norm_category = Category_Preselection
    #workspace_min_clf = 0.


class Category_1J_Inclusive(Category_Preselection):
    name = '1j_inclusive'
    root_label = '#tau_{had}#tau_{had} Inclusive 1-Jet'
    common_cuts = Category_Preselection.common_cuts & CATEGORY_CUTS_MVA
    cuts = AT_LEAST_1JET
    norm_category = Category_Preselection


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
