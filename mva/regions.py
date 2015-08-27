from rootpy.tree import Cut

OS     = Cut('(tau_0_q * tau_1_q) == -1')
NOT_OS = Cut('(tau_0_q * tau_1_q) != -1')
SS     = Cut('(tau_0_q * tau_1_q) ==  1')

P1P1 = Cut('tau_0_n_tracks == 1') & Cut('tau_1_n_tracks == 1')
P3P3 = Cut('tau_0_n_tracks == 3') & Cut('tau_1_n_tracks == 3')
P1P3 = (
    (Cut('tau_0_n_tracks == 1') | Cut('tau_1_n_tracks == 3'))
    &
    (Cut('tau_1_n_tracks == 1') | Cut('tau_0_n_tracks == 3')))

TRACK_ISOLATION = (
    Cut('tau_0_n_wide_tracks == 0')
    & # AND
    Cut('tau_1_n_wide_tracks == 0'))

TRACK_NONISOLATION = (
    Cut('tau_0_n_wide_tracks != 0')
    | # OR
    Cut('tau_1_n_wide_tracks != 0'))

REGIONS = {
    'ALL': Cut(),

    'OS': OS & P1P3,
    'OS_ISOL': OS & P1P3 & TRACK_ISOLATION,
    'OS_NONISOL': OS & P1P3 & TRACK_NONISOLATION,
    'OS_NTRK': OS,

    'SS': SS & P1P3,
    'SS_ISOL': SS & P1P3 & TRACK_ISOLATION,
    'SS_NONISOL': SS & P1P3 & TRACK_NONISOLATION,
    'SS_NTRK': SS,

    'nOS': NOT_OS,
    'nOS_ISOL': NOT_OS & TRACK_ISOLATION,
    'nOS_NONISOL': NOT_OS & TRACK_NONISOLATION,

    'NONISOL': TRACK_NONISOLATION,
}

REGION_SYSTEMATICS = {
    'nOS_NONISOL': 'nOS_ISOL',
    'nOS_ISOL': 'nOS_NONISOL',
    #'nOS': ('nOS_ISOL', 'nOS_NONISOL'),
    'nOS': 'SS',
}
