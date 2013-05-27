import tables
f = tables.openFile('ntuples/prod/HHProcessor/HHProcessor.h5')
t = f.root.McAtNloJimmy_CT10_ttbar_LeptonFilter_mc12a_JES_Detector_DOWN
t.readWhere('(((((~(((jet1_pt>30000)&(~((mmc1_resonance_pt>100)&(~(((dEta_jets>2.0)&((jet1_pt>50000)&(jet2_pt>30000)))&(mmc1_resonance_pt>40))))))&(~(((dEta_jets>2.0)&((jet1_pt>50000)&(jet2_pt>30000)))&(mmc1_resonance_pt>40)))))&(~((mmc1_resonance_pt>100)&(~(((dEta_jets>2.0)&((jet1_pt>50000)&(jet2_pt>30000)))&(mmc1_resonance_pt>40))))))&(~(((dEta_jets>2.0)&((jet1_pt>50000)&(jet2_pt>30000)))&(mmc1_resonance_pt>40))))&(((((((tau1_pt>35000)&(tau2_pt>25000))&(MET>20000))&(mmc1_mass>0))&((0.8<dR_tau1_tau2)&(dR_tau1_tau2<2.8)))&(tau_same_vertex))&(MET_bisecting|(dPhi_min_tau_MET<1.570796))))&(dEta_tau1_tau2<1.5))&(((tau1_charge*tau2_charge)!=-1)&(taus_pass))',
    start=0, step=2)
