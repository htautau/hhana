import math


BDT_BLIND = {
    2015: {
        'vbf': 3,
        'boosted': 1},
    2012: {
        'vbf': 3,
        'boosted': 1},
    2011: {
        'vbf': 2,
        'boosted': 2}
}


def get_label(name, units=True, latex=False):
    info = VARIABLES[name]
    if latex:
        label = info['title']
    else:
        label = info['root']
    if units and 'units' in info:
        label += ' [{0}]'.format(info['units'])
    return label


def get_binning(name, category, year):
    binning = VARIABLES[name]['binning']
    if isinstance(binning, dict):
        if year in binning:
            binning = binning[year]
        if isinstance(binning, dict):
            binning = binning.get(category.name.upper(), binning[None])
    return binning


def get_scale(name):
    info = VARIABLES[name]
    return info.get('scale', 1)


def get_units(name):
    info = VARIABLES[name]
    return info.get('units', None)


def blind_hist(name, hist, year=None, category=None):
    if name.upper() == 'BDT':
        unblind = BDT_BLIND[year][category.name]
        hist[-(unblind + 1):] = (0, 0)
        return
    blind = VARIABLES[name].get('blind', None)
    if blind is None:
        return
    left, right = blind
    left_bin = hist.FindBin(left)
    right_bin = hist.FindBin(right)
    for ibin in xrange(left_bin, right_bin + 1):
        hist[ibin] = (0, 0)


WEIGHTS = {
    'weight_pileup': {
        'title': 'Pile-up Weight',
        'root': 'Pile-up Weight',
        'filename': 'weight_pileup',
        'binning': (50, -.2, 3.)
    },
}

YEAR_VARIABLES = {
    2011: {
        'RunNumber' : {
            'title': r'RunNumber',
            'root': 'Run Number',
            'filename': 'runnumber',
            'binning': [
                177531, 177986, 178163, 179710, 180614,
                182013, 182726, 183544, 185353, 186516,
                186873, 188902, 190503],
        }
    },
    2012: {
        'RunNumber' : {
            'title': r'RunNumber',
            'root': 'Run Number',
            'filename': 'runnumber',
            'binning': [
                200804, 202660, 206248, 207447, 209074, 210184, 211522,
                212619, 213431, 213900, 214281, 215414, 216399],
        }
    }
}

VARIABLES = {
    'n_avg_int': {
        'title': r'$\langle\mu\rangle|_{LB,BCID}$',
        'root': '#font[152]{#LT#mu#GT#cbar}_{LB,BCID}',
        'filename': 'n_avg_int',
        'binning': (40, 0, 40),
        'integer': True,
    },
    'n_actual_int': {
       'title': r'$\langle\mu\rangle|_{LB}(BCID)$',
       'root': '#font[152]{#LT#mu#GT#cbar}_{LB}#font[52]{(BCID)}',
       'filename': 'n_actual_int',
       'binning': (20, 0, 40),
    },
    'n_jets': {
        'title': r'Number of Selected Jets',
        'root': '#font[52]{Number of Selected Jets}',
        'filename': 'n_jets',
        'binning': (10, -.5, 9.5),
        'integer': True,
    },
}

HH_VARIABLES = {
    'ditau_deta': {
        'title': r'$\Delta \Eta_{\tau \tau}$',
        'root': '#font[152]{#Delta #Eta_{#tau #tau}}',
        'filename': 'ditau_deta',
        'binning': (5, 0, 5),
        'units': ''
    },
    'ditau_vect_sum_pt': {
        'title': r'$\sum p_T$ Taus and Two Leading Jets',
        'root': '#font[152]{#sum} #font[52]{p}_{T} #font[52]{Taus and Two Leading Jets}',
        'filename': 'ditau_vect_sum_pt',
        'binning': (20, 50, 550),
        'units': 'GeV',
    },
    'ditau_scal_sum_pt': {
        'title': r'$\sum p_T$ Taus and All Selected Jets',
        'root': '#font[152]{#sum} #font[52]{p}_{T} #font[52]{Taus and All Selected Jets}',
        'filename': 'ditau_scal_sum_pt',
        'binning': (20, 50, 550),
        'units': 'GeV',
    },
    'met_et': {
        'title': r'$E^{miss}_{T}$',
        'root': '#font[52]{E}^{miss}_{T}',
        'filename': 'met_et',
        'binning': {
            'PRESELECTION': (10, 20, 95),
            'REST': (13, 15, 80),
            None: (15, 0, 80)},
        'units': 'GeV',
    },
    'met_etx': {
        'title': r'$E^{miss}_{T_{x}}$',
        'root': '#font[52]{E}^{miss}_{T_{x}}',
        'filename': 'met_etx',
        'binning': (20, -75, 75),
        'units': 'GeV',
        'legend': 'left',
    },
    'met_ety': {
        'title': r'$E^{miss}_{T_{y}}$',
        'root': '#font[52]{E}^{miss}_{T_{y}}',
        'filename': 'met_ety',
        'binning': (20, -75, 75),
        'units': 'GeV',
        'legend': 'left',
    },
    'met_phi': {
        'title': r'$E^{miss}_{T} \phi$',
        'root': '#font[52]{E}^{miss}_{T} #phi',
        'filename': 'met_phi',
        'binning': (5, -math.pi, math.pi),
    },
    'ditau_met_min_dphi': {
        'title': r'min[$\Delta\phi$($\tau$,\/$E^{miss}_{T}$)]',
        'root': '#font[52]{min}[#font[152]{#Delta#phi}(#font[152]{#tau},#font[52]{E}^{miss}_{T})]',
        'filename': 'ditau_met_min_dphi',
        'binning': (10, 0, math.pi),
    },
    'ditau_met_bisect': {
        'title': r'$E^{miss}_{T}$ bisects',
        'root': '#font[52]{E}^{miss}_{T} bisects',
        'filename': 'ditau_met_bisect',
        'binning': (2, -0.5, 1.5),
        'legend': 'left',
        'integer': True,
    },
    'ditau_mt_lep0_met': {
        'title': r'',
        'root': '#font[52]{m}_{T} (l, #font[52]{E}^{miss}_{T})',
        'filename': 'ditau_mt_lep0_met',
        'binning': (20, 0, 100),
        'units': 'GeV',
    },

    'ditau_met_centrality': {
        'title': r'$E^{miss}_{T}$ Centrality',
        'root': '#font[52]{E}^{miss}_{T} #font[152]{#phi #text{Centrality}}',
        'filename': 'ditau_met_centrality',
        'binning': (10, -math.sqrt(2), math.sqrt(2)),
        'legend': 'left',
    },
    'tau1_centrality': {
        'title': r'$#tau_1$ Centrality',
        'root': '#font[52]{#tau_1} #font[152]{#eta #text{Centrality}}',
        'filename': 'tau1_centrality',
        'binning': (10, -math.sqrt(2), math.sqrt(2)),
        'legend': 'left',
    },
    'tau2_centrality': {
        'title': r'$#tau_2$ Centrality',
        'root': '#font[52]{#tau_2} #font[152]{#eta #test{Centrality}}',
        'filename': 'tau2_centrality',
        'binning': (10, -math.sqrt(2), math.sqrt(2)),
        'legend': 'left',
    },

    'ditau_vis_mass': {
        'title': r'$m^{vis}_{\tau\tau}$',
        'root': '#font[52]{m}^{vis}_{#font[152]{#tau}#font[152]{#tau}}',
        'filename': 'ditau_vis_mass',
        'binning': {
            'PRESELECTION': (15, 20, 120),
            'REST': (20, 30, 150),
            None: (20, 0, 250)},
        'units': 'GeV',
        'blind': (70, 110),
    },

    'ditau_coll_approx_m': {
        'title': r'$m^{col}_{\tau\tau}$',
        'root': '#font[52]{m}^{col}_{#font[152]{#tau}#font[152]{#tau}}',
        'filename': 'ditau_coll_approx_m',
        'binning': (20, 0, 250),
        'scale': 0.001,
        'units': 'GeV',
        'blind': (100, 150),
    },
   'ditau_tau0_pt': {
       'title': r'$\tau_{1} p_{T}$',
       'root': '#font[152]{#tau}_{1} #font[52]{p}_{T}',
       'filename': 'ditau_tau0_pt',
       'binning': (7, 30, 100),
       'units': 'GeV',
   },
   'ditau_tau1_pt': {
       'title': r'$\tau_{2} p_{T}$',
       'root': '#font[152]{#tau}_{2} #font[52]{p}_{T}',
       'filename': 'ditau_tau1_pt',
       'binning': (7, 30, 100),
       'units': 'GeV',
   },
   'ditau_tau0_eta': {
       'title': r'$\tau_{1} \eta$',
       'root': '#font[152]{#tau}_{1} #font[152]{#eta}',
       'filename': 'ditau_tau0_eta',
       'binning': [-3.0, -2.5, -1.52, -1.37, -0.6, 0, 0.6, 1.37, 1.52, 2.5, 3.0],
       'legend': 'left',
   },
   'ditau_tau1_eta': {
       'title': r'$\tau_{2} \eta$',
       'root': '#font[152]{#tau}_{2} #font[152]{#eta}',
       'filename': 'ditau_tau1_eta',
       'binning': [-3.0, -2.5, -1.52, -1.37, -0.6, 0, 0.6, 1.37, 1.52, 2.5, 3.0],
       'legend': 'left',
   },
   'eta_product_jets': {
       'title': r'$\eta_{jet1} \times \eta_{jet2}$',
       'root': '#font[152]{#eta}_{j1} #times #font[152]{#eta}_{j2}',
       'filename': 'eta_product_jets',
       'binning': [-12.0, -2.5, -1.52, -1.37, -0.6, 0, 0.6, 1.37, 1.52, 2.5, 12.0],
       'legend': 'left',
   },

   'ditau_tau0_n_tracks': {
       'title': r'$\tau_{1}$ Number of Tracks',
       'root': '#font[152]{#tau}_{1} #font[52]{Tracks}',
       'filename': 'ditau_tau0_n_tracks',
       'binning': (5, -.5, 4.5),
       'integer': True,
   },
   'ditau_tau1_n_tracks': {
       'title': r'$\tau_{2}$ Number of Tracks',
       'root': '#font[152]{#tau}_{2} #font[52]{Tracks}',
       'filename': 'ditau_tau1_n_tracks',
       'binning': (5, -.5, 4.5),
       'integer': True,
   },

   # not yet in the ntuples
   # 'tau1_numTrack_recounted': {
   #     'title': r'$\tau_{1}$ Number of Recounted Tracks',
   #     'root': '#font[152]{#tau}_{1} #font[52]{Recounted Tracks}',
   #     'filename': 'tau1_numTrack_recounted',
   #     'binning': (5, 0.5, 5.5),
   #     'integer': True,
   # },
   # 'tau2_numTrack_recounted': {
   ##     'title': r'$\tau_{2}$ Number of Recounted Tracks',
   #    'root': '#font[152]{#tau}_{2} #font[52]{Recounted Tracks}',
   #     'filename': 'tau2_numTrack_recounted',
   #     'binning': (5, 0.5, 5.5),
   #     'integer': True,
   # },

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
    #'tau1_collinear_momentum_fraction': {
    #    'title': r'$\tau_{1}$ Visible Momentum Fraction',
    #    'root': '#font[152]{#tau}_{x1}',
    #    'filename': 'tau1_x',
    #    'binning': (20, -3, 4),
    #},
    #'tau2_collinear_momentum_fraction': {
    #    'title': r'$\tau_{2}$ Visible Momentum Fraction',
    #    'root': '#font[152]{#tau}_{x2}',
    #    'filename': 'tau2_x',
    #    'binning': (20, -3, 4),
    #},

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
   'ditau_tau0_jet_bdt_score': {
      'title': r'$\tau_{1}$ BDT Score',
      'root': '#font[152]{#tau}_{1} #font[52]{BDT Score}',
      'filename': 'ditau_tau0_jet_bdt_score',
      'binning': (5, 0.5, 1.0001),
   },
   'ditau_tau1_jet_bdt_score': {
      'title': r'$\tau_{2}$ BDT Score',
      'root': '#font[152]{#tau}_{2} #font[52]{BDT Score}',
      'filename': 'ditau_tau1_jet_bdt_score',
      'binning': (5, 0.5, 1.0001),
   },
    #'tau1_vertex_prob': {
    #    'title': r'$\tau_{1}$ Primary Vertex Quality',
    #    'root': '#tau_{1} Primary Vertex Quality',
    #    'filename': 'tau1_vertex_quality',
    #    'bins': 20,
    #    'range': (-0.0001, 1.0001),
    #    'cats': ['1J', '2J',]
    #},
   'ditau_cosalpha': {
       'title': r'$\cos[\alpha(\tau,\tau)]$',
       'root': '#font[52]{cos}(#font[152]{#alpha}_{#font[152]{#tau}#font[152]{#tau}})',
       'filename': 'ditau_cosalpha',
       'binning': (12, -1.2, 1.2),
   },
   'ditau_dr': {
       'title': r'$\Delta R(\tau,\tau)$',
       'root': '#font[152]{#Delta}#font[52]{R}(#font[152]{#tau},#font[152]{#tau})',
       'filename': 'ditau_dr',
       'binning': {
           None: (20, 0, 5),
           'PRESELECTION': (9, 0.4, 2.8)},
       'ypadding': (0.5, 0),
   },
   'ditau_dphi': {
       'title': r'$\Delta \phi(\tau,\tau)$',
       'root': '#font[152]{#Delta#phi}(#font[152]{#tau},#font[152]{#tau})',
       'filename': 'ditau_dphi',
       'binning': (6, 0., 2.4),
       'legend': 'left',
   },
   'dEta_jets': {
       'title': r'$\Delta \eta(\tau,\tau)$',
       'root': '#font[152]{#Delta#eta}(#font[152]{#tau},#font[152]{#tau})',
       'filename': 'dEta_jets',
       'binning': {
           'BOOSTED': (10, 0, 1.5),
           'VBF': (10, 0, 1.5),
           'REST': (10, 0, 1.5),
           None: (8, 0, 2.0)},
       'ypadding': (0.5, 0),
   },
   'ditau_tau0_q': {
      'title': r'$\tau_1$ Charge',
      'root': '#font[152]{#tau}_{1} #font[52]{Charge}',
      'filename': 'ditau_tau0_q',
      'binning': (7, -3.5, 3.5),
      'integer': True,
   },
   'ditau_tau1_q': {
      'title': r'$\tau_2$ Charge',
      'root': '#font[152]{#tau}_{2} #font[52]{Charge}',
      'filename': 'ditau_tau1_q',
      'binning': (7, -3.5, 3.5),
      'integer': True,
   },
   'jet_1_eta': {
       'title': r'jet$_{2}$ $\eta$',
       'root': '#font[152]{#eta}(#font[52]{j}2)',
       'filename': 'jet_1_eta',
       'binning': (20, -5, 5),
       'cats': ['2J', 'VBF', '1J', '1J_NONBOOSTED'],
       'legend': 'left',
   },
   'jet_0_eta': {
       'title': r'jet$_{1}$ $\eta$',
       'root': '#font[152]{#eta}(#font[52]{j}1)',
       'filename': 'jet_0_eta',
       'binning': (20, -5, 5),
       'cats': ['2J', 'VBF'],
       'legend': 'left',
   },
   'jet_1_pt': {
       'title': r'jet$_{2}$ $p_{T}$',
       'root': '#font[52]{p}_{T}(#font[52]{j}2)',
       'filename': 'jet_1_pt',
       'binning': (10, 15, 80),
       'scale': 1,
       'units': 'GeV',
       'cats': ['2J', 'VBF', '1J', '1J_NONBOOSTED']
   },
   'jet_0_pt': {
       'title': r'jet$_{0}$ $p_{T}$',
       'root': '#font[52]{p}_{T}(#font[52]{j}1)',
       'filename': 'jet_0_pt',
       'binning': (10, 15, 80),
       'scale': 1,
       'units': 'GeV',
       'cats': ['2J', 'VBF']
   },
#    'HCM2': {
#        'title': r'HCM2',
#        'root': '#font[52]HCM2',
#        'filename': 'HCM2',
#        'binning': {
#            'BOOSTED': (0.01, 0.0, 1.0),
#            'VBF': (0.01, 0.0, 1.0),
#            'REST': (0.01, 0.0, 1.0),
#            None: (0.01, 0.0, 1.0)},
#        'scale': 0.01,
#        'units': 'GeV',
#    },
    'HCM1': {
        'title': r'HCM1',
        'root': '#font[52]{HCM1}',
        'filename': 'HCM1',
        'binning': {
            'BOOSTED': (15, 0.0, 1.0),
            'VBF': (15, 0.0, 1.0),
            'REST': (15, 0.0, 1.0),
            None: (20, 0.0, 1.0)},
        'scale': 1,
        'units': 'No Units',
    },
    'HCM2': {
        'title': r'HCM2',
        'root': '#font[52]{HCM2}',
        'filename': 'HCM2',
        'binning': {
            'BOOSTED': (15, 0.0, 1.0),
            'VBF': (15, 0.0, 1.0),
            'REST': (15, 0.0, 1.0),
            None: (20, 0.0, 1.0)},
        'scale': 1,
        'units': 'No Units'
    },
    'HCM3': {
        'title': r'HCM3',
        'root': '#font[52]{HCM3}',
        'filename': 'HCM3',
        'binning': {
            'BOOSTED': (15, 0.0, 1.0),
            'VBF': (15, 0.0, 1.0),
            'REST': (15, 0.0, 1.0),
            None: (20, 0.0, 1.0)},
        'scale': 1,
        'units': 'No Units',
    },
    'HCM2jj': {
        'title': r'HCM2jj',
        'root': '#font[52]{HCM2jj}',
        'filename': 'HCM2jj',
        'binning': {
            'BOOSTED': (15, 0.0, 1.0),
            'VBF': (15, 0.0, 1.0),
            'REST': (15, 0.0, 1.0),
            None: (20, 0.0, 1.0)},
        'scale': 1,
        'units': 'No Units',
    },

    'mass_jet1_jet2': {
        'title': r'$m^{vis}_{jj}$',
        'root': '#font[52]{m}_{#font[52]{j}#font[52]{j}}',
        'filename': 'mass_jet1_jet2',
        'binning': (20, 0, 800),
        'units': 'GeV',
        'scale': 0.001,
    },

    'ditau_pt_ratio': {
        'title': r'$\tau_{1} p_{T} / \tau_{2} p_{T}$',
        'root': '#font[52]{p}_{T}(#font[152]{#tau}_{1}) / #font[52]{p}_{T}(#font[152]{#tau}_{2})',
        'filename': 'ditau_pt_ratio',
        'binning': (16, 1, 5),
    },

}

LH_VARIABLES = {
    # LEPHAD STUFF
    'lep_0_pt':{
        'title': r'p_T(l)',
        'root': '#font[52]{p}_{T}(l)',
        'filename': 'lep_0_pt',
        'binning': (20, 20, 120),
        'scale': 0.001,
        'units': 'GeV',
        },
    'met_reco_et': {
        'title': r'$E^{miss}_{T}$',
        'root': '#font[52]{E}^{miss}_{T}',
        'filename': 'met_reco_et',
        'binning': {
            'PRESELECTION': (13, 15, 80),
            'REST': (13, 15, 80),
            None: (15, 0, 80)},
        'scale': 0.001,
        'units': 'GeV',
    },

#    'mass_tau1_tau2_jet1': {
#        'title': r'$m^{vis}_{j\tau\tau}$',
#        'root': '#font[52]{m}_{#font[52]{j}#font[152]{#tau}#font[152]{#tau}}',
#        'filename': 'mass_taus_leading_jet',
#        'binning': (20, 0, 800),
#        'units': 'GeV',
#        'scale': 0.001,
#    },
#    'jet3_centrality': {
#        'title': r'j3 Centrality',
#        'root': '#font[52]{j}_{3} #font[152]{#eta} centrality',
#        'filename': 'jet3_centrality',
#        'binning': (20, 0, 1),
#        'cats': ['2J', 'VBF'],
#        'legend': 'left',
#    },

    'pt_ratio_lep_tau': {
        'title': r'$\tau_{1} p_{T} / \tau_{2} p_{T}$',
        'root': '#font[52]{p}_{T}(#font[152]{#tau}_{1}) / #font[52]{p}_{T}(#font[152]{#tau}_{2})',
        'filename': 'pt_ratio_lep_tau',
        'binning': (16, 1, 5),
    },
     'tau_0_pt': {
         'title': r'$\tau p_{T}$',
         'root': '#font[152]{#tau} #font[52]{p}_{T}',
         'filename': 'tau_0_pt',
         'binning': {
             2011: {
                 'PRESELECTION': (10, 35, 90),
                 'REST': (10, 35, 90),
                 None: (10, 35, 160)},
             2012: {
                 'PRESELECTION': (20, 35, 90),
                 'REST': (20, 35, 90),
                 None: (20, 35, 160)},
             2015: {
                 'PRESELECTION': (20, 35, 90),
                 'REST': (20, 35, 90),
                 None: (15, 20, 100)}},
         'scale': 0.001,
         'units': 'GeV',
         },
    'tau_0_n_tracks': {
        'title': r'$\tau$ Number of Tracks',
        'root': '#font[152]{#tau} #font[52]{Tracks}',
        'filename': 'tau_0_n_tracks',
        'binning': (5, -.5, 4.5),
    },
    'jet_0_pt': {
        'title': r'jet$_{0}$ $p_{T}$',
        'root': '#font[52]{p}_{T}(#font[52]{j}1)',
        'filename': 'jet_0_pt',
        'binning': (10, 15, 80),
        'scale': 0.001,
        'units': 'GeV',
        'cats': ['2J', 'VBF']
    },
    'jet_0_eta': {
        'title': r'jet$_{1}$ $\eta$',
        'root': '#font[152]{#eta}(#font[52]{j}1)',
        'filename': 'jet_0_eta',
        'binning': (20, -5, 5),
        'cats': ['2J', 'VBF'],
        'legend': 'left',
    },
    'lephad_mmc_mlm_m': {
        'title': r'$m^{MMC}_{\tau\tau}$',
        'root': '#font[52]{m}^{MMC}_{#font[152]{#tau}#font[152]{#tau}}',
        'filename': 'lephad_mmc_mlm_m',
        'binning': (20, 0, 250),
        'units': 'GeV',
        'scale': 0.001,
        'blind': (100, 150),
    },
    'lephad_coll_approx_m': {
        'title': r'$m^{col}_{\tau\tau}$',
        'root': '#font[52]{m}^{col}_{#font[152]{#tau}#font[152]{#tau}}',
        'filename': 'lephad_coll_approx_m',
        'binning': (20, 0, 250),
        'units': 'GeV',
        'scale': 0.001,
        'blind': (100, 150),
    },
    'lephad_met_centrality': {
        'title': r'$E^{miss}_{T}$ Centrality',
        'root': '#font[52]{E}^{miss}_{T} #font[152]{#phi Centrality}',
        'filename': 'lephad_met_centrality',
        'binning': (20, -math.sqrt(2), math.sqrt(2)),
        'legend': 'left',
    },
    'lephad_dr': {
        'title': r'$\Delta R(\tau,\tau)$',
        'root': '#font[152]{#Delta}#font[52]{R}(#font[152]{#tau},#font[152]{#tau})',
        'filename': 'lephad_dr',
        'binning': (20, 0, 5),
        'ypadding': (0.5, 0),
    },
    'lephad_dphi': {
        'title': r'$\Delta \phi(\tau,\tau)$',
        'root': '#font[152]{#Delta#phi}(#font[152]{#tau},#font[152]{#tau})',
        'filename': 'lephad_dphi',
        'binning': (12, 0., 2.4),
        'legend': 'left',
    },
    'lephad_deta': {
        'title': r'$\Delta \eta(\tau,\tau)$',
        'root': '#font[152]{#Delta#eta}(#font[152]{#tau},#font[152]{#tau})',
        'filename': 'lephad_deta',
        'binning': {
            'BOOSTED': (10, 0, 1.5),
            'VBF': (10, 0, 1.5),
            'REST': (10, 0, 1.5),
            None: (10, 0, 2.5)},
        'ypadding': (0.5, 0),
    },
    'jets_delta_eta': {
        'title': r'$\Delta\eta(jet_{1},\/jet_{2})$',
        'root': '#font[152]{#Delta#eta}(#font[52]{j}_{1},#font[52]{j}_{2})',
        'filename': 'jets_delta_eta',
        'cuts': 'dEta_jets > 0', # ignore default value in plot
        'binning': {
            'VBF': (10, 2, 7),
            None: (14, 0, 7)},
        'cats': ['2J', 'VBF', 'PRESELECTION']
    },
    'prod_eta_jets': {
        'title': r'jet$_{1}$ $\eta \times \/$ jet$_{2}$ $\eta$',
        'root': '#font[152]{#eta}_{#font[52]{j}_{1}} #times #font[152]{#eta}_{#font[52]{j}_{2}}',
        'filename': 'prod_eta_jets',
        'binning': (15, -10, 5),
        'cats': ['2J', 'VBF'],
        'legend': 'left',
    },
     'jets_visible_mass': {
        'title': r'$M(jet_{1},\/jet_{2})$',
        'root': '#font[52]{m}_{#font[52]{j}#font[52]{j}}',
        'filename': 'jets_visible_mass',
        'binning': (20, 0, 1000),
        'scale': 0.001,
        'units': 'GeV',
        'cats': ['2J', 'VBF']
    },


}


from . import MMC_MASS

HH_VARIABLES[MMC_MASS] = {
    'title': r'$m^{MMC}_{\tau\tau}$',
    'root': '#font[52]{m}^{MMC}_{#font[152]{#tau}#font[152]{#tau}}',
    'filename': MMC_MASS,
    'binning': {
        2011: (25, 0, 250),
        2012: (25, 0, 250),
        2015: (13, 0, 260)},
    'units': 'GeV',
    'blind': (100, 150),
}

"""
VARIABLES['mmc%d_MET_et' % mmc] = {
    'title': r'$E^{miss}_{T}$ MMC',
    'root': '#font[52]{MMC} #font[52]{E}^{miss}_{T}',
    'filename': 'mmc%d_MET' % mmc,
    'binning': (20, 0, 100),
    'units': 'GeV',
}

VARIABLES['mmc%d_MET_etx' % mmc] = {
    'title': r'MMC $E^{miss}_{T_{x}}$',
    'root': '#font[52]{MMC} #font[52]{E}^{miss}_{T_{x}}',
    'filename': 'mmc%d_MET_x' % mmc,
    'binning': (20, -75, 75),
    'units': 'GeV',
}

VARIABLES['mmc%d_MET_ety' % mmc] = {
    'title': r'MMC $E^{miss}_{T_{y}}$',
    'root': '#font[52]{MMC} #font[52]{E}^{miss}_{T_{y}}',
    'filename': 'mmc%d_MET_y' % mmc,
    'binning': (20, -75, 75),
    'units': 'GeV',
}

VARIABLES['mmc%d_resonance_pt' % mmc] = {
    'title': r'MMC $p_T^H$',
    'root': 'MMC #font[52]{p}_{T}^{H}',
    'filename': 'mmc%d_resonance_pt' % mmc,
    'binning': {
        'BOOSTED': (20, 50, 200),
        None: (20, 0, 200)},
    'units': 'GeV',
}
"""
