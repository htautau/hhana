from .. import MMC_MASS

features_vbf = [
    MMC_MASS,
    'dEta_jets',
    'eta_product_jets',
    'mass_jet1_jet2',
    #'sphericity',
    #'aplanarity',
    'tau1_centrality',
    'tau2_centrality',
    'dR_tau1_tau2',
    #'tau1_BDTJetScore',
    #'tau2_BDTJetScore',
    'MET_centrality',
    'vector_sum_pt',
    #'sum_pt_full',
    #'resonance_pt',
    #'jet3_centrality',
]

features_boosted = [
    MMC_MASS,
    #'mass_tau1_tau2_jet1',
    #'sphericity',
    #'aplanarity',
    'dR_tau1_tau2',
    #'tau1_BDTJetScore',
    #'tau2_BDTJetScore',
    'tau1_collinear_momentum_fraction',
    'tau2_collinear_momentum_fraction',
    'MET_centrality',
    #'resonance_pt',
    #'jet1_pt',
    'sum_pt_full',
    'tau_pt_ratio',
]
