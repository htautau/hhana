from .. import MMC_MASS

features_vbf = [
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
