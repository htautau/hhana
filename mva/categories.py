from rootpy.tree import Cut

from .samples import Higgs

TAU1_MEDIUM = Cut('tau1_JetBDTSigMedium==1')
TAU2_MEDIUM = Cut('tau2_JetBDTSigMedium==1')
TAU1_TIGHT = Cut('tau1_JetBDTSigTight==1')
TAU2_TIGHT = Cut('tau2_JetBDTSigTight==1')

TAU1_CENTRAL = Cut('fabs(tau1_eta) < 1.5')
TAU1_FORWARD = Cut('fabs(tau1_eta) > 1.5')
TAU2_CENTRAL = Cut('fabs(tau2_eta) < 1.5')
TAU2_FORWARD = Cut('fabs(tau2_eta) > 1.5')

ID_MEDIUM = TAU1_MEDIUM & TAU2_MEDIUM
ID_TIGHT = TAU1_TIGHT & TAU2_TIGHT
ID_MEDIUM_TIGHT = (TAU1_MEDIUM & TAU2_TIGHT) | (TAU1_TIGHT & TAU2_MEDIUM)

Z_PEAK = Cut('60 < mass_mmc_tau1_tau2 < 120')
ID_MEDIUM_FORWARD_TIGHT_CENTRAL = (
        (TAU1_MEDIUM & TAU1_FORWARD & TAU2_TIGHT & TAU2_CENTRAL) |
        (TAU1_TIGHT & TAU1_CENTRAL & TAU2_MEDIUM & TAU2_FORWARD))

TAU_DR_CUT = Cut('dR_tau1_tau2 < 3.2')
TAU_DETA_CUT = Cut('dEta_tau1_tau2 < 1.5')
TAU_SAME_VERTEX = Cut('tau_same_vertex')
BAD_MASS = 60
MASS_FIX = Cut('mass_mmc_tau1_tau2 > %d' % BAD_MASS)
MAX_NJET = Cut('numJets <= 3')
MET = Cut('MET > 20000')

LEAD_TAU_35 = Cut('tau1_pt > 35000')
SUBLEAD_TAU_25 = Cut('tau2_pt > 25000')

COMMON_CUTS = (
        LEAD_TAU_35 & SUBLEAD_TAU_25 &
        MET & MASS_FIX &
        TAU_DR_CUT & TAU_DETA_CUT &
        TAU_SAME_VERTEX)

LEAD_JET_50 = Cut('jet1_pt > 50000')
SUBLEAD_JET_30 = Cut('jet2_pt > 30000')
AT_LEAST_1JET = Cut('jet1_pt > 30000')

CUTS_2J = LEAD_JET_50 & SUBLEAD_JET_30
CUTS_1J = LEAD_JET_50 & (- SUBLEAD_JET_30)
CUTS_0J = (- LEAD_JET_50)

CUTS_VBF = Cut('dEta_jets > 2.0')
CUTS_BOOSTED = Cut('mmc_resonance_pt > 100') # GeV

# TODO: possible new variable: ratio of core tracks to recounted tracks
# TODO: add new pi0 info (new variables?)


features_2j = [
    'mass_mmc_tau1_tau2',
    # !!! mass ditau + leading jet?
    'dEta_jets',
    #'dEta_jets_boosted', #
    'eta_product_jets',
    #'eta_product_jets_boosted', #
    'mass_jet1_jet2',
    #'sphericity', #
    #'sphericity_boosted', #
    #'sphericity_full', #
    #'aplanarity', #
    #'aplanarity_boosted', #
    #'aplanarity_full', #
    'tau1_centrality',
    'tau2_centrality',
    #'tau1_centrality_boosted', #
    #'tau2_centrality_boosted', #
    #'cos_theta_tau1_tau2', #
    'dR_tau1_tau2',
    #'tau1_BDTJetScore',
    #'tau2_BDTJetScore',
    #'tau1_x', #
    #'tau2_x', #
    'MET_centrality',
    'mmc_resonance_pt',
    #'sum_pt', #
    # !!! eta centrality of 3rd jet
]

features_1j = [
    'mass_mmc_tau1_tau2',
    # !!! mass ditau + leading jet?
    'sphericity',
    #'sphericity_boosted',
    #'sphericity_full',
    #'aplanarity',
    #'aplanarity_boosted',
    #'aplanarity_full',
    #'cos_theta_tau1_tau2',
    'dR_tau1_tau2',
    #'tau1_BDTJetScore',
    #'tau2_BDTJetScore',
    'tau1_x',
    'tau2_x',
    'MET_centrality',
    #'sum_pt',
    'mmc_resonance_pt',
]

features_0j = [
    'mass_mmc_tau1_tau2',
    #'cos_theta_tau1_tau2',
    'dR_tau1_tau2',
    #'tau1_BDTJetScore',
    #'tau2_BDTJetScore',
    'tau1_x',
    'tau2_x',
    'MET_centrality',
    'mmc_resonance_pt',
]


class Category(object):

    # common attrs for all categories. Override in subclasses
    year_cuts = {
        2011: ID_MEDIUM,
        2012: ID_MEDIUM_TIGHT}
    qcd_shape_region = 'SS'
    target_region = 'OS'
    train_signal_modes = Higgs.MODES[:]


# Default categories

class Category_2J(Category):

    name = '2j'
    label = r'$\tau_{had}\tau_{had}$: 2-Jet Category'
    cuts = CUTS_2J & COMMON_CUTS
    fitbins = 5
    limitbins = 8
    limitbinning = 'onebkg'
    features = features_2j


class Category_1J(Category):

    name = '1j'
    label = r'$\tau_{had}\tau_{had}$: 1-Jet Category'
    cuts = CUTS_1J & COMMON_CUTS
    fitbins = 5
    limitbins = 10
    limitbinning = 'onebkg'
    features = features_1j


class Category_0J(Category):

    name = '0j'
    label = r'$\tau_{had}\tau_{had}$: 0-Jet Category'
    cuts = CUTS_0J & COMMON_CUTS
    fitbins = 8
    limitbins = 10
    limitbinning = 'onebkg'
    features = features_0j


# Harmonization

class Category_VBF(Category):

    name = 'vbf'
    label = r'$\tau_{had}\tau_{had}$: VBF Category'
    cuts = CUTS_VBF & CUTS_2J & COMMON_CUTS
    fitbins = 5
    limitbins = 8
    limitbinning = 'onebkg'
    features = features_2j
    # train with only VBF
    signal_train_modes = ['VBF']


class Category_Boosted(Category):

    name = 'boosted'
    label = r'$\tau_{had}\tau_{had}$: Boosted Category'
    cuts = CUTS_BOOSTED & (- (CUTS_VBF & CUTS_2J)) & COMMON_CUTS
    fitbins = 5
    limitbins = 10
    limitbinning = 'onebkg'
    # warning: some variables will be undefined for some events
    features = features_2j
    # train with all modes


class Category_Nonboosted_1J(Category):

    name = '1j_nonboosted'
    label = r'$\tau_{had}\tau_{had}$: Non-boosted 1-Jet Category'
    cuts = AT_LEAST_1JET & (- (CUTS_BOOSTED & (- (CUTS_VBF & CUTS_2J)))) & COMMON_CUTS
    fitbins = 5
    limitbins = 10
    limitbinning = 'onebkg'
    features = features_1j
    # train with all modes


class Category_Nonboosted_0J(Category):

    name = '0j_nonboosted'
    label = r'$\tau_{had}\tau_{had}$: Non-boosted 0-Jet Category'
    cuts = - (AT_LEAST_1JET & (- (CUTS_BOOSTED & (- (CUTS_VBF & CUTS_2J))))) & COMMON_CUTS
    fitbins = 8
    limitbins = 10
    limitbinning = 'onebkg'
    features = features_0j
    # train with all modes


CATEGORIES = {
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
    ]
}

CONTROLS = {
    'preselection': {
        'name': r'$\tau_{had}\tau_{had}$: At Preselection',
        'cuts': COMMON_CUTS,
        'year_cuts': {
            2011: ID_MEDIUM,
            2012: ID_MEDIUM_TIGHT},
        'fitbins': 10,
        'qcd_shape_region': 'SS',
        'target_region': 'OS',
    },
    'z': {
        'name': r'$\tau_{had}\tau_{had}$: Z Control Region',
        'cuts': MET & Cut('dR_tau1_tau2<2.8') & Z_PEAK,
        'year_cuts': {
            2011: ID_MEDIUM,
            2012: ID_MEDIUM_TIGHT},
        'fitbins': 8,
        'qcd_shape_region': 'SS',
        'target_region': 'OS',
    }
}
