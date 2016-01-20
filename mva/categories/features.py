from .. import MMC_MASS

features_vbf = [
#    MMC_MASS,
    'dEta_jets',
    'eta_product_jets',
    'mass_jet1_jet2',
    #'sphericity',
    #'aplanarity',
#    'tau1_centrality',
#    'tau2_centrality',
    'ditau_dr',#'dR_tau1_tau2', Changed name in Run II
    #'tau1_BDTJetScore',
    #'tau2_BDTJetScore',
    'ditau_met_centrality',
    'ditau_vect_sum_pt',
    #'sum_pt_full',
    #'resonance_pt',
    #'jet3_centrality',
#    'HCM2jj',
#    'HCM2',
#    'HCM1',
    'HCM3'
]

features_boosted = [
#    MMC_MASS,
    #'mass_tau1_tau2_jet1',
    #'sphericity',
    #'aplanarity',
    'ditau_dr',#'dR_tau1_tau2', Changed name in Run II
    #'tau1_BDTJetScore',
    #'tau2_BDTJetScore',
    #'tau1_collinear_momentum_fraction',
    #'tau2_collinear_momentum_fraction',
    'ditau_met_centrality',
    #'resonance_pt',
    #'jet1_pt',
    'ditau_scal_sum_pt',
    'ditau_pt_ratio',
#    'HCM2',
#    'HCM1',
    'HCM3',
]
