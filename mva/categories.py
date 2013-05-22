from rootpy.tree import Cut
import math
from . import MMC_MASS, MMC_PT

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
CUTS_BOOSTED = Cut('%s > 100' % MMC_PT) # GeV

BAD_MASS = 60

COMMON_CUTS_CUTBASED = (
    LEAD_TAU_35 & SUBLEAD_TAU_25
    #& MET <= no MET cut in cut-based preselection
    & Cut('%s > 0' % MMC_MASS)
    & Cut('0.8 < dR_tau1_tau2 < 2.8')
    & TAU_SAME_VERTEX
    & Cut('MET_bisecting || (dPhi_min_tau_MET < (0.2 * %f))' % math.pi)
    )

COMMON_CUTS_MVA = (
    LEAD_TAU_35 & SUBLEAD_TAU_25
    & Cut('MET > 20000')
    & Cut('%s > %d' % (MMC_MASS, BAD_MASS))
    & Cut('0.8 < dR_tau1_tau2 < 2.8')
    & TAU_SAME_VERTEX
    )

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
    #MMC_PT,
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
    'tau1_x', #  <= ADD BACK IN
    'tau2_x', #  <= ADD BACK IN
    'MET_centrality',
    #MMC_PT,
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
    'tau1_x',
    'tau2_x',
    'MET_centrality',
    'sum_pt_full',
    'tau_pt_ratio',
    #MMC_PT,
]

features_0j = [
    MMC_MASS,
    #'cos_theta_tau1_tau2',
    'dR_tau1_tau2',
    #'tau1_BDTJetScore',
    #'tau2_BDTJetScore',
    'tau1_x',
    'tau2_x',
    'MET_centrality',
    'sum_pt_full',
    'tau_pt_ratio',
    #MMC_PT,
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
    is_control = False
    # category used for normalization
    norm_category = None
    qcd_shape_region = 'nOS' # no track cut
    target_region = 'OS_TRK'
    cuts = Cut()
    common_cuts = Cut()
    from . import samples
    train_signal_modes = samples.Higgs.MODES[:]
    clf_bins = 10
    # only unblind up to this number of bins in half-blind mode
    halfblind_bins = 0

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


# Cut-based categories

class Category_Cuts_Preselection(Category):

    name = 'cut_preselection'
    label = r'$\tau_{had}\tau_{had}$: At Cut-based Preselection'
    common_cuts = COMMON_CUTS_CUTBASED


class Category_Cuts_Preselection_DEta_Control(Category_Cuts_Preselection):

    name = 'cut_preselection_deta_control'
    label = r'$\tau_{had}\tau_{had}$: $\Delta \eta_{\tau_{1},\/\tau_{2}} \geq 1.5$ Control Region at Cut-based Preselection'


# MVA preselection categories

class Category_Preselection(Category):

    name = 'preselection'
    label = r'$\tau_{had}\tau_{had}$: At Preselection'
    common_cuts = COMMON_CUTS_MVA
    cuts = Cut('theta_tau1_tau2 > 0.6')


class Category_Preselection_ID_Control(Category_Preselection):

    is_control = True
    name = 'preselection_id_control'
    label = r'$\tau_{had}\tau_{had}$: ID Control Region at Preselection'


class Category_Preselection_DEta_Control(Category_Preselection):

    is_control = True
    name = 'preselection_deta_control'
    label = r'$\tau_{had}\tau_{had}$: $\Delta \eta_{\tau_{1},\/\tau_{2}} \geq 1.5$ Control Region at Preselection'


# Default categories

class Category_2J(Category_Preselection):

    name = '2j'
    label = r'$\tau_{had}\tau_{had}$: 2-Jet Category'
    cuts = CUTS_2J
    fitbins = 5
    limitbins = 8
    limitbinning = 'onebkg'
    features = features_2j


class Category_1J(Category_Preselection):

    name = '1j'
    label = r'$\tau_{had}\tau_{had}$: 1-Jet Category'
    cuts = CUTS_1J
    fitbins = 5
    limitbins = 10
    limitbinning = 'onebkg'
    features = features_1j


class Category_0J(Category_Preselection):

    name = '0j'
    label = r'$\tau_{had}\tau_{had}$: 0-Jet Category'
    cuts = CUTS_0J
    fitbins = 8
    limitbins = 10
    limitbinning = 'onebkg'
    features = features_0j


# Harmonization

class Category_VBF(Category_Preselection):

    name = 'vbf'
    label = r'$\tau_{had}\tau_{had}$: VBF Category'
    cuts = CUTS_VBF & CUTS_2J
    fitbins = 5
    limitbins = 8
    limitbinning = 'onebkg'
    features = features_2j
    # train with only VBF
    signal_train_modes = ['VBF']
    halfblind_bins = 3

    norm_category = Category_Preselection


class Category_VBF_ID_Control(Category_VBF):

    is_control = True
    name = 'vbf_id_control'
    label = r'$\tau_{had}\tau_{had}$: VBF Category ID Control Region'


class Category_VBF_DEta_Control(Category_VBF):

    is_control = True
    name = 'vbf_deta_control'
    label = r'$\tau_{had}\tau_{had}$: VBF Category $\Delta \eta_{\tau_{1},\/\tau_{2}} \geq 1.5$ Control Region'


class Category_Boosted(Category_Preselection):

    name = 'boosted'
    label = r'$\tau_{had}\tau_{had}$: Boosted Category'
    cuts = CUTS_BOOSTED & (- Category_VBF.cuts)
    fitbins = 5
    limitbins = 10
    limitbinning = 'onebkg'
    # warning: some variables will be undefined for some events
    features = features_boosted
    # train with all modes

    halfblind_bins = 2

    norm_category = Category_Preselection


class Category_Boosted_ID_Control(Category_Boosted):

    is_control = True
    name = 'boosted_id_control'
    label = r'$\tau_{had}\tau_{had}$: Boosted Category ID Control Region'
    year_cuts = {
        2011: ID_MEDIUM_NOT_TIGHT,
        2012: ID_MEDIUM_NOT_TIGHT}


class Category_Boosted_DEta_Control(Category_Boosted):

    is_control = True
    name = 'boosted_deta_control'
    label = r'$\tau_{had}\tau_{had}$: Boosted Category $\Delta \eta_{\tau_{1},\/\tau_{2}} \geq 1.5$ Control Region'



class Category_Nonboosted_1J(Category_Preselection):

    name = '1j_nonboosted'
    label = r'$\tau_{had}\tau_{had}$: Non-boosted 1-Jet Category'
    cuts = AT_LEAST_1JET & (- Category_Boosted.cuts) & (- Category_VBF.cuts)
    fitbins = 5
    limitbins = 10
    limitbinning = 'onebkg'
    features = features_1j
    # train with all modes

    halfblind_bins = 5

    norm_category = Category_Preselection


class Category_Nonboosted_1J_ID_Control(Category_Nonboosted_1J):

    is_control = True
    name = '1j_nonboosted_id_control'
    label = r'$\tau_{had}\tau_{had}$: Non-boosted 1-Jet Category ID Control Region'


class Category_Nonboosted_1J_DEta_Control(Category_Nonboosted_1J):

    is_control = True
    name = '1j_nonboosted_deta_control'
    label = r'$\tau_{had}\tau_{had}$: Non-boosted 1-Jet Category $\Delta \eta_{\tau_{1},\/\tau_{2}} \geq 1.5$ Control Region'


class Category_Nonboosted_0J(Category_Preselection):

    name = '0j_nonboosted'
    label = r'$\tau_{had}\tau_{had}$: Non-boosted 0-Jet Category'
    cuts = (- Category_Nonboosted_1J.cuts) & (- Category_Boosted.cuts) & (- Category_VBF.cuts)
    fitbins = 8
    limitbins = 10
    limitbinning = 'onebkg'
    features = features_0j
    # train with all modes

    halfblind_bins = 5

    norm_category = Category_Preselection


class Category_Nonboosted_0J_ID_Control(Category_Nonboosted_0J):

    is_control = True
    name = '0j_nonboosted_id_control'
    label = r'$\tau_{had}\tau_{had}$: Non-boosted 0-Jet Category ID Control Region'


class Category_Nonboosted_0J_DEta_Control(Category_Nonboosted_0J):

    is_control = True
    name = '0j_nonboosted_deta_control'
    label = r'$\tau_{had}\tau_{had}$: Non-boosted 0-Jet Category $\Delta \eta_{\tau_{1},\/\tau_{2}} \geq 1.5$ Control Region'


CATEGORIES = {
    'cuts_presel': [
        Category_Cuts_Preselection,
        ],
    'cuts_presel_id_controls': [
        ],
    'cuts_presel_deta_controls': [
        Category_Cuts_Preselection_DEta_Control,
        ],
    'presel': [
        Category_Preselection,
        ],
    'presel_id_controls': [
        ],
    'presel_deta_controls': [
        ],
    'default': [
        Category_2J,
        Category_1J,
        Category_0J,
        ],
    'harmonize': [
        Category_VBF,
        Category_Boosted,
        Category_Nonboosted_1J,
        Category_Nonboosted_0J,
    ],
    'harmonize_id_controls': [
        Category_VBF_ID_Control,
        Category_Boosted_ID_Control,
        Category_Nonboosted_1J_ID_Control,
        Category_Nonboosted_0J_ID_Control,
    ],
    'harmonize_deta_controls': [
        Category_VBF_DEta_Control,
        Category_Boosted_DEta_Control,
        Category_Nonboosted_1J_DEta_Control,
        Category_Nonboosted_0J_DEta_Control,
    ]
}
