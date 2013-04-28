from rootpy.tree import Cut

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
    'OS': OS,
    'OS_TRK': OS & P1P3_RECOUNTED,
    'nOS': NOT_OS,
    'SS': SS,
    'SS_TRK': SS & P1P3_RECOUNTED}

TARGET_REGIONS = ['OS', 'OS_TRK']
QCD_SHAPE_REGIONS = ['nOS', 'SS', 'SS_TRK']
