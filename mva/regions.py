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

TRACK_NONISOLATION = (
    Cut('tau1_numTrack_recounted > tau1_numTrack')
    |
    Cut('tau2_numTrack_recounted > tau2_numTrack'))


REGIONS = {
    'ALL': Cut(),
    'OS': OS & P1P3 & TRACK_ISOLATION,
    'nOS': NOT_OS & TRACK_ISOLATION,
    'SS': SS & P1P3 & TRACK_ISOLATION,
    'NONISOL': TRACK_NONISOLATION,
}

TARGET_REGIONS = ['OS',]
QCD_SHAPE_REGIONS = ['nOS', 'SS', 'NONISOL',]
