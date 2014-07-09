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
    & # AND
    Cut('tau2_numTrack_recounted == tau2_numTrack'))

TRACK_NONISOLATION = (
    Cut('tau1_numTrack_recounted > tau1_numTrack')
    | # OR
    Cut('tau2_numTrack_recounted > tau2_numTrack'))

REGIONS = {
    'ALL': Cut(),

    'OS': OS & P1P3,
    'OS_ISOL': OS & P1P3 & TRACK_ISOLATION,
    'OS_NONISOL': OS & P1P3 & TRACK_NONISOLATION,

    'SS': SS & P1P3,
    'SS_ISOL': SS & P1P3 & TRACK_ISOLATION,
    'SS_NONISOL': SS & P1P3 & TRACK_NONISOLATION,

    'nOS': NOT_OS,
    'nOS_ISOL': NOT_OS & TRACK_ISOLATION,
    'nOS_NONISOL': NOT_OS & TRACK_NONISOLATION,

    'NONISOL': TRACK_NONISOLATION,
}

REGION_SYSTEMATICS = {
    'nOS_NONISOL': 'nOS_ISOL',
    'nOS_ISOL': 'nOS_NONISOL',
    'nOS': 'SS',
    'SS': 'nOS',
}
