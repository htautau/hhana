from rootpy.tree import Cut

OS = Cut('(tau1_charge * tau2_charge) == -1')
NOT_OS = Cut('(tau1_charge * tau2_charge) != -1')
SS = Cut('(tau1_charge * tau2_charge) == 1')

P1P1 = Cut('tau1_numTrack == 1') & Cut('tau2_numTrack == 1')
P3P3 = Cut('tau1_numTrack == 3') & Cut('tau2_numTrack == 3')

P1P3 = (
    (Cut('tau1_numTrack == 1') | Cut('tau1_numTrack == 3'))
    &
    (Cut('tau2_numTrack == 1') | Cut('tau2_numTrack == 3')))

TRACK_ISOLATION = (
    Cut('tau1_numTrack_recounted == tau1_numTrack')
    &
    Cut('tau2_numTrack_recounted == tau2_numTrack'))

REGIONS = {
    'ALL': Cut(),
    'OS': OS & TRACK_ISOLATION,
    'OS_TRK': OS & P1P3 & TRACK_ISOLATION,
    'nOS': NOT_OS & TRACK_ISOLATION,
    'SS': SS & TRACK_ISOLATION,
    'SS_TRK': SS & P1P3 & TRACK_ISOLATION,
}

TARGET_REGIONS = ['OS', 'OS_TRK']
QCD_SHAPE_REGIONS = [
    'nOS', 'SS', 'SS_TRK',
]
