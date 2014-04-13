import math

def get_label(name):
    info = VARIABLES[name]
    label = info['root']
    if 'units' in info:
        label += ' [{0}]'.format(info['units'])
    return label

WEIGHTS = {
    'pileup_weight': {
        'title': 'Pile-up Weight',
        'root': 'Pile-up Weight',
        'filename': 'weight_pileup',
        'bins': 50,
        'range': (-.2, 3.)
    },
}

YEAR_VARIABLES = {
    2011: {
        'RunNumber' : {
            'title': r'RunNumber',
            'root': 'Run Number',
            'filename': 'runnumber',
            'bins': (
                177531,
                177986,
                178163,
                179710,
                180614,
                182013,
                182726,
                183544,
                185353,
                186516,
                186873,
                188902,
                190503),
        }
    },
    2012: {
        'RunNumber' : {
            'title': r'RunNumber',
            'root': 'Run Number',
            'filename': 'runnumber',
            'bins': (
                200804, 202660, 206248, 207447, 209074, 210184, 211522,
                212619, 213431, 213900, 214281, 215414, 216399),
        }
    }
}

VARIABLES = {
    #'ntrack_pv': {
    #    'title': r'Number of Tracks from the Primary Vertex',
    #    'root': '#font[52]{Number of Tracks from the Primary Vertex}',
    #    'filename': 'ntrack_pv',
    #    'bins': 10,
    #    'range': (0.5, 120.5),
    #},
    #'ntrack_nontau_pv': {
    #    'title': r'Number of Non-Tau Tracks from the Primary Vertex',
    #    'root': '#font[52]{Number of Non-Tau Tracks from the Primary Vertex}',
    #    'filename': 'ntrack_nontau_pv',
    #    'bins': 10,
    #    'range': (0.5, 120.5),
    #},
    'averageIntPerXing': {
        'title': r'$\langle\mu\rangle|_{LB,BCID}$',
        'root': '#font[152]{#LT#mu#GT#cbar}_{LB,BCID}',
        'filename': 'averageIntPerXing',
        'bins': 20,
        'range': (0, 40),
    },
    #'actualIntPerXing': {
    #    'title': r'$\langle\mu\rangle|_{LB}(BCID)$',
    #    'root': '#font[152]{#LT#mu#GT#cbar}_{LB}#font[52]{(BCID)}',
    #    'filename': 'actualIntPerXing',
    #    'bins': 20,
    #    'range': (0, 40),
    #},
    'sum_pt': {
        'title': r'$\sum p_T$ Taus and Two Leading Jets',
        'root': '#font[152]{#sum} #font[52]{p}_{T} #font[52]{Taus and Two Leading Jets}',
        'filename': 'sum_pt',
        'bins': 20,
        'range': (50, 550),
        'scale': 0.001,
        'units': 'GeV',
    },
    'sum_pt_full': {
        'title': r'$\sum p_T$ Taus and All Selected Jets',
        'root': '#font[152]{#sum} #font[52]{p}_{T} #font[52]{Taus and All Selected Jets}',
        'filename': 'sum_pt_full',
        'bins': 20,
        'range': (50, 550),
        'scale': 0.001,
        'units': 'GeV',
    },
    'vector_sum_pt': {
        'title': r'$\sum \vec{p}_T$',
        'root': '#font[52]{p}_{T}^{Total}',
        'filename': 'vector_sum_pt',
        'bins': 20,
        'range': {
            'VBF': (0, 100),
            None: (0, 200)},
        'scale': 0.001,
        'units': 'GeV',
    },
    'numJets': {
        'title': r'Number of Selected Jets',
        'root': '#font[52]{Number of Selected Jets}',
        'filename': 'numjets',
        'bins': 7,
        'range': (-.5, 6.5),
        'integer': True,
    },
    'MET_et': {
        'title': r'$E^{miss}_{T}$',
        'root': '#font[52]{E}^{miss}_{T}',
        'filename': 'MET',
        'bins': 20,
        'range': {
            'PRESELECTION': (13, 15, 80),
            'REST': (13, 15, 80),
            None: (17, 15, 100)},
        'scale': 0.001,
        'units': 'GeV',
    },
    'MET_etx': {
        'title': r'$E^{miss}_{T_{x}}$',
        'root': '#font[52]{E}^{miss}_{T_{x}}',
        'filename': 'MET_x',
        'bins': 20,
        'range': (-75, 75),
        'scale': 0.001,
        'units': 'GeV',
    },
    'MET_ety': {
        'title': r'$E^{miss}_{T_{y}}$',
        'root': '#font[52]{E}^{miss}_{T_{y}}',
        'filename': 'MET_y',
        'bins': 20,
        'range': (-75, 75),
        'scale': 0.001,
        'units': 'GeV',
    },
    'MET_phi': {
        'title': r'$E^{miss}_{T} \phi$',
        'root': '#font[52]{E}^{miss}_{T} #phi',
        'filename': 'MET_phi',
        'bins': 20,
        'range': (-math.pi, math.pi),
    },
    'dPhi_min_tau_MET': {
        'title': r'min[$\Delta\phi$($\tau$,\/$E^{miss}_{T}$)]',
        'root': '#font[52]{min}[#font[152]{#Delta#phi}(#font[152]{#tau},#font[52]{E}^{miss}_{T})]',
        'filename': 'dPhi_min_tau_MET',
        'bins': 20,
        'range': (0, math.pi),
    },
    'MET_bisecting': {
        'title': r'MET bisects the taus',
        'root': '#font[52]{E}^{miss}_{T} bisects the taus',
        'filename': 'MET_bisecting',
        'bins': 2,
        'range': (-0.5, 1.5),
        'legend': 'left',
        'integer': True,
    },
    #'sphericity': {
    #    'title': r'Sphericity',
    #    'root': '#font[52]{Sphericity}',
    #    'filename': 'sphericity',
    #    'bins': 20,
    #    'range': (0, 1),
    #    'cats': ['2J', 'VBF', '1J', '1J_NONBOOSTED'],
    #},
    #'aplanarity': {
    #    'title': r'Aplanarity',
    #    'root': '#font[52]{Aplanarity}',
    #    'filename': 'aplanarity',
    #    'bins': 20,
    #    'range': (0, .15),
    #    'cats': ['2J', 'VBF', '1J', '1J_NONBOOSTED'],
    #},
    'MET_centrality': {
        'title': r'$E^{miss}_{T}$ Centrality',
        'root': '#font[52]{E}^{miss}_{T} #font[152]{#phi} centrality',
        'filename': 'met_centrality',
        'bins': 20,
        'range': (-math.sqrt(2), math.sqrt(2)),
        'legend': 'left',
    },
    'mass_vis_tau1_tau2': {
        'title': r'$M^{vis}(\tau_{1},\/\tau_{2})$',
        'root': '#font[52]{m}^{vis}_{#font[152]{#tau}#font[152]{#tau}}',
        'filename': 'mass_vis',
        'bins': 20,
        'range': {
            'PRESELECTION': (30, 150),
            'REST': (30, 150),
            None: (0, 250)},
        'scale': 0.001,
        'units': 'GeV',
        'blind': (70, 110),
    },
    'mass_collinear_tau1_tau2': {
        'title': r'$M^{col}(\tau_{1},\/\tau_{2})$',
        'root': '#font[52]{m}^{col}_{#font[152]{#tau}#font[152]{#tau}}',
        'filename': 'mass_collinear',
        'bins': 20,
        'range': (0, 250),
        'units': 'GeV',
        'scale': 0.001,
        'blind': (100, 150),
    },
    'tau_pt_ratio': {
        'title': r'$\tau_{1} p_{T} / \tau_{2} p_{T}$',
        'root': '#font[52]{p}_{T}(#font[152]{#tau}_{1}) / #font[52]{p}_{T}(#font[152]{#tau}_{2})',
        'filename': 'tau_pt_ratio',
        'bins': 20,
        'range': (0, 5),
    },
    'tau1_pt': {
        'title': r'$\tau_{1} p_{T}$',
        'root': '#font[152]{#tau}_{1} #font[52]{p}_{T}',
        'filename': 'tau1_pt',
        'bins': 20,
        'range': {
            'PRESELECTION': (35, 90),
            'REST': (35, 90),
            None: (35, 160)},
        'scale': 0.001,
        'units': 'GeV',
    },
    'tau2_pt': {
        'title': r'$\tau_{2} p_{T}$',
        'root': '#font[152]{#tau}_{2} #font[52]{p}_{T}',
        'filename': 'tau2_pt',
        'bins': 20,
        'range': {
            'PRESELECTION': (25, 60),
            'REST': (25, 60),
            None: (25, 100)},
        'scale': 0.001,
        'units': 'GeV',
    },
    'tau1_eta': {
        'title': r'$\tau_{1} \eta$',
        'root': '#font[152]{#tau}_{1} #font[152]{#eta}',
        'filename': 'tau1_eta',
        'bins': 20,
        'range': (-3, 3),
    },
    'tau2_eta': {
        'title': r'$\tau_{2} \eta$',
        'root': '#font[152]{#tau}_{2} #font[152]{#eta}',
        'filename': 'tau2_eta',
        'bins': 20,
        'range': (-3, 3),
    },
    'tau1_numTrack': {
        'title': r'$\tau_{1}$ Number of Tracks',
        'root': '#font[152]{#tau}_{1} #font[52]{Number of Tracks}',
        'filename': 'tau1_numTrack',
        'bins': 5,
        'range': (-.5, 4.5),
    },
    'tau2_numTrack': {
        'title': r'$\tau_{2}$ Number of Tracks',
        'root': '#font[152]{#tau}_{2} #font[52]{Number of Tracks}',
        'filename': 'tau2_numTrack',
        'bins': 5,
        'range': (-.5, 4.5),
        'integer': True,
    },
    'tau1_numTrack_recounted': {
        'title': r'$\tau_{1}$ Number of Recounted Tracks',
        'root': '#font[152]{#tau}_{1} #font[52]{Number of Recounted Tracks}',
        'filename': 'tau1_numTrack_recounted',
        'bins': 6,
        'range': (-.5, 5.5),
        'integer': True,
    },
    'tau2_numTrack_recounted': {
        'title': r'$\tau_{2}$ Number of Recounted Tracks',
        'root': '#font[152]{#tau}_{2} #font[52]{Number of Recounted Tracks}',
        'filename': 'tau2_numTrack_recounted',
        'bins': 6,
        'range': (-.5, 5.5),
        'integer': True,
    },
    #'tau1_nPi0': {
    #    'title': r'$\tau_{1}$ Number of $\pi^0$s',
    #    'root': '#font[152]{#tau}_{1} #font[52]{Number of} #font[152]{#pi}^{0}#font[52]{s}',
    #    'filename': 'tau1_npi0',
    #    'bins': 7,
    #    'range': (-.5, 6.5),
    #    'integer': True,
    #},
    #'tau2_nPi0': {
    #    'title': r'$\tau_{2}$ Number of $\pi^0$s',
    #    'root': '#font[152]{#tau}_{2} #font[52]{Number of} #font[152]{#pi}^{0}#font[52]{s}',
    #    'filename': 'tau2_npi0',
    #    'bins': 7,
    #    'range': (-.5, 6.5),
    #    'integer': True,
    #},
    #'tau_x_product': {
    #    'title': r'Product of $\tau$ Visible Momentum Fractions',
    #    'root': 'Product of #font[152]{#tau} #font[52]{Visible Momentum Fractions}',
    #    'filename': 'tau_x_product',
    #    'bins': 20,
    #    'range': (-9, 16),
    #},
    #'tau_x_sum': {
    #    'title': r'Sum of $\tau$ Visible Momentum Fractions',
    #    'root': 'Sum of #font[152]{#tau} #font[52]{Visible Momentum Fractions}',
    #    'filename': 'tau_x_sum',
    #    'bins': 20,
    #    'range': (-6, 8),
    #},
    'tau1_collinear_momentum_fraction': {
        'title': r'$\tau_{1}$ Visible Momentum Fraction',
        'root': '#font[152]{#tau}_{x1}',
        'filename': 'tau1_x',
        'bins': 20,
        'range': (-3, 4),
    },
    'tau2_collinear_momentum_fraction': {
        'title': r'$\tau_{2}$ Visible Momentum Fraction',
        'root': '#font[152]{#tau}_{x2}',
        'filename': 'tau2_x',
        'bins': 20,
        'range': (-3, 4),
    },
    #'tau1_jvtxf': {
    #    'title': r'$\tau_{1}$ JVF',
    #    'root': '#font[152]{#tau}_{1} #font[52]{JVF}',
    #    'filename': 'tau1_jvf',
    #    'bins': 20,
    #    'range': (0, 1),
    #},
    #'tau2_jvtxf': {
    #    'title': r'$\tau_{2}$ JVF',
    #    'root': '#font[152]{#tau}_{2} #font[52]{JVF}',
    #    'filename': 'tau2_jvf',
    #    'bins': 20,
    #    'range': (0, 1),
    #},
    #'tau1_BDTJetScore': {
    #    'title': r'$\tau_{1}$ BDT Score',
    #    'root': '#font[152]{#tau}_{1} #font[52]{BDT Score}',
    #    'filename': 'tau1_BDTJetScore',
    #    'bins': 20,
    #    'range': (.55, 1.0001),
    #},
    #'tau2_BDTJetScore': {
    #    'title': r'$\tau_{2}$ BDT Score',
    #    'root': '#font[152]{#tau}_{2} #font[52]{BDT Score}',
    #    'filename': 'tau2_BDTJetScore',
    #    'bins': 20,
    #    'range': (.55, 1.0001),
    #},
    #'tau1_vertex_prob': {
    #    'title': r'$\tau_{1}$ Primary Vertex Quality',
    #    'root': '#tau_{1} Primary Vertex Quality',
    #    'filename': 'tau1_vertex_quality',
    #    'bins': 20,
    #    'range': (-0.0001, 1.0001),
    #    'cats': ['1J', '2J',]
    #},
    'cos_theta_tau1_tau2': {
        'title': r'$\cos[\alpha(\tau_{1},\/\tau_{2})]$',
        'root': '#font[52]{cos}(#font[152]{#alpha}_{#font[152]{#tau}#font[152]{#tau}})',
        'filename': 'cos_theta_tau1_tau2',
        'bins': 20,
        'range': (-1, 1),
    },
    'theta_tau1_tau2': {
        'title': r'$\alpha(\tau_{1},\/\tau_{2})$',
        'root': '#font[152]{#alpha}_{#font[152]{#tau}#font[152]{#tau}}',
        'filename': 'theta_tau1_tau2',
        'bins': 20,
        'range': (0, math.pi),
    },
    'dR_tau1_tau2': {
        'title': r'$\Delta R(\tau_{1},\/\tau_{2})$',
        'root': '#font[152]{#Delta}#font[52]{R}(#font[152]{#tau},#font[152]{#tau})',
        'filename': 'dr_tau1_tau2',
        'bins': 8,
        'range': (0.8, 2.4),
        'ypadding': (0.5, 0),
    },
    'dPhi_tau1_tau2': {
        'title': r'$\Delta \phi(\tau_{1},\/\tau_{2})$',
        'root': '#font[152]{#Delta#phi}(#font[152]{#tau},#font[152]{#tau})',
        'filename': 'dphi_tau1_tau2',
        'bins': 12,
        'range': (0., 2.4),
        'legend': 'left',
    },
    'dEta_tau1_tau2': {
        'title': r'$\Delta \eta(\tau_{1},\/\tau_{2})$',
        'root': '#font[152]{#Delta#eta}(#font[152]{#tau},#font[152]{#tau})',
        'filename': 'deta_tau1_tau2',
        'bins': 10,
        'range': {
            'BOOSTED': (10, 0, 1.5),
            'VBF': (10, 0, 1.5),
            'REST': (10, 0, 1.5),
            None: (10, 0, 2.5)},
        'ypadding': (0.5, 0),
    },
    #'tau1_charge': {
    #    'title': r'$\tau_1$ Charge',
    #    'root': '#font[152]{#tau}_{1} #font[52]{Charge}',
    #    'filename': 'tau1_charge',
    #    'bins': 5,
    #    'range': (-2.5, 2.5),
    #    'integer': True,
    #},
    #'tau2_charge': {
    #    'title': r'$\tau_2$ Charge',
    #    'root': '#font[152]{#tau}_{2} #font[52]{Charge}',
    #    'filename': 'tau2_charge',
    #    'bins': 5,
    #    'range': (-2.5, 2.5),
    #    'integer': True,
    #},
    #'tau1_seedCalo_centFrac': {
    #    'title': r'$\tau_1$ Centrality Fraction',
    #    'root': '#font[152]{#tau}_{1} #font[52]{Centrality Fraction}',
    #    'filename': 'tau1_centfrac',
    #    'bins': 20,
    #    'range': (0.5, 1),
    #},
    #'tau_centrality_product': {
    #    'title': r'$\tau$ Centrality Product',
    #    'root': '#font[152]{#tau} centrality product',
    #    'filename': 'tau_centrality_product',
    #    'bins': 20,
    #    'range': (0, 1),
    #    'cats': ['2J', 'VBF']
    #},
    'tau1_centrality': {
        'title': r'$\tau_1$ Centrality',
        'root': '#font[152]{#tau}_{1} #font[152]{#eta} centrality',
        'filename': 'tau1_centrality',
        'bins': 20,
        'range': (0, 1),
        'cats': ['2J', 'VBF'],
        'legend': 'left',
    },
    'tau2_centrality': {
        'title': r'$\tau_2$ Centrality',
        'root': '#font[152]{#tau}_{2} #font[152]{#eta} centrality',
        'filename': 'tau2_centrality',
        'bins': 20,
        'range': (0, 1),
        'cats': ['2J', 'VBF'],
        'legend': 'left',
    },
    'jet1_eta': {
        'title': r'jet$_{1}$ $\eta$',
        'root': '#font[152]{#eta}(#font[52]{j}1)',
        'filename': 'jet1_eta',
        'bins': 20,
        'range': (-5, 5),
        'cats': ['2J', 'VBF', '1J', '1J_NONBOOSTED']
    },
    'jet2_eta': {
        'title': r'jet$_{2}$ $\eta$',
        'root': '#font[152]{#eta}(#font[52]{j}2)',
        'filename': 'jet2_eta',
        'bins': 20,
        'range': (-5, 5),
        'cats': ['2J', 'VBF']
    },
    'jet1_pt': {
        'title': r'jet$_{1}$ $p_{T}$',
        'root': '#font[52]{p}_{T}(#font[52]{j}1)',
        'filename': 'jet1_pt',
        'bins': 20,
        'range': (20, 200),
        'scale': 0.001,
        'units': 'GeV',
        'cats': ['2J', 'VBF', '1J', '1J_NONBOOSTED']
    },
    'jet2_pt': {
        'title': r'jet$_{2}$ $p_{T}$',
        'root': '#font[52]{p}_{T}(#font[52]{j}2)',
        'filename': 'jet2_pt',
        'bins': 20,
        'range': (20, 200),
        'scale': 0.001,
        'units': 'GeV',
        'cats': ['2J', 'VBF']
    },
    'dEta_jets': {
        'title': r'$\Delta\eta(jet_{1},\/jet_{2})$',
        'root': '#font[152]{#Delta#eta}(#font[52]{j}_{1},#font[52]{j}_{2})',
        'filename': 'dEta_jets',
        'cuts': 'dEta_jets > 0', # ignore default value in plot
        'bins': 15,
        'range': {
            'VBF': (15, 0, 7.5),
            None: (15, 0, 7.5)},
        'cats': ['2J', 'VBF', 'PRESELECTION']
    },
    'eta_product_jets': {
        'title': r'jet$_{1}$ $\eta \times \/$ jet$_{2}$ $\eta$',
        'root': '#font[152]{#eta}_{#font[52]{j}_{1}} #times #font[152]{#eta}_{#font[52]{j}_{2}}',
        'filename': 'eta_product_jets',
        'bins': 15,
        'range': (-10, 5),
        'cats': ['2J', 'VBF'],
        'legend': 'left',
    },
    #'eta_product_jets_boosted': {
    #    'title': r'Boosted $\eta_{jet_{1}} \times \/ \eta_{jet_{2}}$',
    #    'root': 'Boosted #eta_{jet_{1}} #times #eta_{jet_{2}}',
    #    'filename': 'eta_product_jets_boosted',
    #    'bins': 20,
    #    'range': (-10, 10),
    #    'cats': ['2J', 'VBF']
    #},
    'mass_jet1_jet2': {
        'title': r'$M(jet_{1},\/jet_{2})$',
        'root': '#font[52]{m}_{#font[52]{j}#font[52]{j}}',
        'filename': 'mass_jet1_jet2',
        'bins': 20,
        'range': (0, 1000),
        'scale': 0.001,
        'units': 'GeV',
        'cats': ['2J', 'VBF']
    },
    'resonance_pt': {
        'title': r'$p_T^{H}$',
        'root': '#font[52]{p}_{T}^{H}',
        'filename': 'resonance_pt',
        'bins': 20,
        'range': {
            'BOOSTED': (18, 70, 250),
            'VBF': (25, 0, 250),
            'REST': (11, 0, 110),
            None: (20, 0, 200)},
        'scale': 0.001,
        'units': 'GeV',
    },
}

for mmc in range(2):

    VARIABLES['mmc%d_mass' % mmc] = {
        'title': r'$M^{MMC}(\tau_{1},\/\tau_{2})$',
        'root': '#font[52]{m}^{MMC}_{#font[152]{#tau}#font[152]{#tau}}',
        'filename': 'mmc%d_mass' % mmc,
        'bins': 25,
        'range': (0, 250),
        'units': 'GeV',
        'blind': (100, 150),
    }

    VARIABLES['mmc%d_MET_et' % mmc] = {
        'title': r'$E^{miss}_{T}$ MMC',
        'root': '#font[52]{MMC} #font[52]{E}^{miss}_{T}',
        'filename': 'mmc%d_MET' % mmc,
        'bins': 20,
        'range': (0, 100),
        'units': 'GeV',
    }

    VARIABLES['mmc%d_MET_etx' % mmc] = {
        'title': r'MMC $E^{miss}_{T_{x}}$',
        'root': '#font[52]{MMC} #font[52]{E}^{miss}_{T_{x}}',
        'filename': 'mmc%d_MET_x' % mmc,
        'bins': 20,
        'range': (-75, 75),
        'units': 'GeV',
    }

    VARIABLES['mmc%d_MET_ety' % mmc] = {
        'title': r'MMC $E^{miss}_{T_{y}}$',
        'root': '#font[52]{MMC} #font[52]{E}^{miss}_{T_{y}}',
        'filename': 'mmc%d_MET_y' % mmc,
        'bins': 20,
        'range': (-75, 75),
        'units': 'GeV',
    }

    VARIABLES['mmc%d_resonance_pt' % mmc] = {
        'title': r'MMC $p_T^H$',
        'root': 'MMC #font[52]{p}_{T}^{H}',
        'filename': 'mmc%d_resonance_pt' % mmc,
        'bins': 20,
        'range': {'BOOSTED': (50, 200), None: (0, 200)},
        'units': 'GeV',
    }
