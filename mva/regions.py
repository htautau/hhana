from rootpy.tree import Cut
# Import does not work :-/
# from categories.hadhad import ID_MEDIUM, ANTI_ID_MEDIUM

OS     = Cut('(tau_0_q * tau_1_q) < 0')
NOT_OS = -OS #Cut('(tau_0_q * tau_1_q) != -1')
SS     = Cut('(tau_0_q * tau_1_q) >  0')

P1P1 = Cut('tau_0_n_tracks == 1') & Cut('tau_1_n_tracks == 1')
P3P3 = Cut('tau_0_n_tracks == 3') & Cut('tau_1_n_tracks == 3')
P1P3 = (
    (Cut('tau_0_n_tracks == 1') | Cut('tau_0_n_tracks == 3'))
    &
    (Cut('tau_1_n_tracks == 1') | Cut('tau_1_n_tracks == 3')))

TRACK_ISOLATION = (
    Cut('tau_0_n_wide_tracks == 0')
    & # AND
    Cut('tau_1_n_wide_tracks == 0'))

TRACK_NONISOLATION = (
    Cut('tau_0_n_wide_tracks != 0')
    | # OR
    Cut('tau_1_n_wide_tracks != 0'))


TAU1_LOOSE = Cut('tau_0_jet_bdt_loose==1')
TAU1_MEDIUM = Cut('tau_0_jet_bdt_medium==1')
TAU1_ANTI_MEDIUM = TAU1_LOOSE & -TAU1_MEDIUM

TAU2_LOOSE = Cut('tau_1_jet_bdt_loose==1')
TAU2_MEDIUM = Cut('tau_1_jet_bdt_medium==1')
TAU2_ANTI_MEDIUM = TAU2_LOOSE & -TAU2_MEDIUM

ID_MEDIUM = TAU1_MEDIUM & TAU2_MEDIUM
ANTI_ID_MEDIUM = TAU1_ANTI_MEDIUM & TAU2_ANTI_MEDIUM



REGIONS = {
    'ALL': Cut(),

    'OS': OS & P1P3 & ID_MEDIUM,
    'OS_ISOL': OS & P1P3 & ID_MEDIUM & TRACK_ISOLATION,
    'OS_NONISOL': OS & P1P3 & ID_MEDIUM & TRACK_NONISOLATION,

    'SS': SS & P1P3 & ID_MEDIUM,
    'SS_ISOL': SS & P1P3 & ID_MEDIUM & TRACK_ISOLATION,
    'SS_NONISOL': SS & P1P3 & ID_MEDIUM & TRACK_NONISOLATION,

    'nOS': NOT_OS & ID_MEDIUM,
    'nOS_ISOL': NOT_OS & ID_MEDIUM & TRACK_ISOLATION,
    'nOS_NONISOL': NOT_OS & ID_MEDIUM & TRACK_NONISOLATION,

    'OS_NONID': OS & P1P3 & ANTI_ID_MEDIUM & TRACK_ISOLATION,
    'OS_NONID_NONISOL': OS & P1P3 & ANTI_ID_MEDIUM & TRACK_NONISOLATION,

    'NONISOL': TRACK_NONISOLATION,
}

REGION_SYSTEMATICS = {
    'nOS_NONISOL': 'nOS_ISOL',
    'nOS_ISOL': 'nOS_NONISOL',
    #'nOS': ('nOS_ISOL', 'nOS_NONISOL'),
    'nOS': 'SS',
}
