from rootpy.tree import Cut

TAUS_PASS = Cut('taus_pass')
TAUS_FAIL = Cut('!taus_pass')
OS = Cut('(tau1_charge * tau2_charge) == -1')
NOT_OS = Cut('(tau1_charge * tau2_charge) != -1')
SS = Cut('(tau1_charge * tau2_charge) == 1')

P1P3 = (
    (Cut('tau1_numTrack == 1') | Cut('tau1_numTrack == 3'))
    &
    (Cut('tau2_numTrack == 1') | Cut('tau2_numTrack == 3')))

P1P3_RECOUNTED = (
    (Cut('tau1_numTrack_recounted == 1') | Cut('tau1_numTrack_recounted == 3'))
    &
    (Cut('tau2_numTrack_recounted == 1') | Cut('tau2_numTrack_recounted == 3')))

REGIONS = {
    'ALL': Cut(),
    'OS': OS & TAUS_PASS,
    'OSFF': OS & TAUS_FAIL,
    'OS_TRK': OS & TAUS_PASS & P1P3_RECOUNTED,
    'OSFF_TRK': OS & TAUS_FAIL & P1P3_RECOUNTED,
    'nOS': NOT_OS & TAUS_PASS,
    'nOSFF': NOT_OS & TAUS_FAIL,
    'SS': SS & TAUS_PASS,
    'SSFF': SS & TAUS_FAIL,
    'SS_TRK': SS & TAUS_PASS & P1P3_RECOUNTED,
    'SSFF_TRK': SS & TAUS_FAIL & P1P3_RECOUNTED}

TARGET_REGIONS = ['OS', 'OS_TRK']
QCD_SHAPE_REGIONS = ['nOS', 'SS', 'SS_TRK', 'OSFF', 'SSFF', 'SSFF_TRK']
