from rootpy.tree import Cut

TRUE_RESONANCE_PT = Cut('true_resonance_pt>100000')
TRUE_LEAD_JET_50 = Cut('true_jet1_no_overlap_pt>50000')
TRUE_SUBLEAD_JET_30 = Cut('true_jet2_no_overlap_pt>30000')
TRUE_2J = Cut('num_true_jets_no_overlap>1')
TRUE_JETS_DETA = 'true_dEta_jet1_jet2_no_overlap>{0}'
TRUE_JETS_MASS = Cut('true_mass_jet1_jet2_no_overlap>250000')

CUTS_TRUE_VBF = (
    TRUE_2J 
    & TRUE_LEAD_JET_50
    & TRUE_SUBLEAD_JET_30
    & Cut(TRUE_JETS_DETA.format(2.0))
    )

CUTS_TRUE_BOOSTED = (
    TRUE_RESONANCE_PT
    )

CUTS_TRUE_VBF_CUTBASED = (
    TRUE_2J 
    & TRUE_LEAD_JET_50
    & TRUE_SUBLEAD_JET_30
    & Cut(TRUE_JETS_DETA.format(2.6))
    & TRUE_JETS_MASS
    )    
