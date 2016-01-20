from rootpy.tree import Cut

# Import does not work :-/
# from categories.hadhad import ID_MEDIUM, ANTI_ID_MEDIUM

Q = (
    (Cut('ditau_tau0_q == 1') | Cut('ditau_tau0_q == -1'))
    &
    (Cut('ditau_tau1_q == 1') | Cut('ditau_tau1_q == -1'))
    )

OS     = Cut('selection_opposite_sign==1')
NOT_OS = -OS
SS     = Cut('ditau_qxq >  1')

P1P1 = Cut('ditau_tau0_n_tracks == 1') & Cut('ditau_tau1_n_tracks == 1')
P3P3 = Cut('ditau_tau0_n_tracks == 3') & Cut('ditau_tau1_n_tracks == 3')
P1P3 = (
    (Cut('ditau_tau0_n_tracks == 1') | Cut('ditau_tau0_n_tracks == 3'))
    &
    (Cut('ditau_tau1_n_tracks == 1') | Cut('ditau_tau1_n_tracks == 3')))

TRACK_ISOLATION = (
    Cut('ditau_tau0_n_wide_tracks == 0')
    & # AND
    Cut('ditau_tau1_n_wide_tracks == 0'))

TRACK_NONISOLATION = (
    Cut('ditau_tau0_n_wide_tracks != 0')
    | # OR
    Cut('ditau_tau1_n_wide_tracks != 0'))


TAU1_LOOSE = Cut('ditau_tau0_jet_bdt_loose==1')
TAU1_MEDIUM = Cut('ditau_tau1_jet_bdt_medium==1')
TAU1_TIGHT = Cut('ditau_tau0_jet_bdt_tight==1')
TAU1_ANTI_MEDIUM = -TAU1_MEDIUM & TAU1_LOOSE

TAU2_LOOSE = Cut('ditau_tau1_jet_bdt_loose==1')
TAU2_MEDIUM = Cut('ditau_tau1_jet_bdt_medium==1')
TAU2_TIGHT = Cut('ditau_tau1_jet_bdt_tight==1')
TAU2_ANTI_MEDIUM = -TAU2_MEDIUM & TAU2_LOOSE

ID_MEDIUM = TAU1_MEDIUM & TAU2_MEDIUM
ANTI_ID_MEDIUM = TAU1_ANTI_MEDIUM & TAU2_ANTI_MEDIUM

ID_TIGHT = TAU1_TIGHT & TAU2_TIGHT
ID_MEDIUM_TIGHT = (TAU1_MEDIUM & TAU2_TIGHT) | (TAU1_TIGHT & TAU2_MEDIUM)
# ID cuts for control region where both taus are medium but not tight
ID_MEDIUM_NOT_TIGHT = (TAU1_MEDIUM & -TAU1_TIGHT) & (TAU2_MEDIUM & -TAU2_TIGHT)


REGIONS = {
    'ALL': Cut(),

    'OS': OS & P1P3 & ID_MEDIUM & Q,
    'OS_ISOL': OS & P1P3 & ID_MEDIUM & TRACK_ISOLATION & Q,
    'OS_NONISOL': OS & P1P3 & ID_MEDIUM & TRACK_NONISOLATION & Q,

    'SS': SS & P1P3 & ID_MEDIUM & Q,
    'SS_ISOL': SS & P1P3 & ID_MEDIUM & TRACK_ISOLATION & Q,
    'SS_NONISOL': SS & P1P3 & ID_MEDIUM & TRACK_NONISOLATION & Q,
    'SS_NONID_ISOL': SS & P1P3 & ANTI_ID_MEDIUM & Q,

    'nOS': NOT_OS & ID_MEDIUM & Q,
    'nOS_ISOL': NOT_OS & ID_MEDIUM & TRACK_ISOLATION & Q,
    'nOS_NONISOL': NOT_OS & ID_MEDIUM & TRACK_NONISOLATION & Q,

    'OS_NONID': OS & P1P3 & ANTI_ID_MEDIUM & Q,
    'OS_NONID_1': OS & P1P3 & TAU1_ANTI_MEDIUM & Q,
    'OS_NONID_2': OS & P1P3 & TAU2_ANTI_MEDIUM & Q,
    'OS_NONID_ISOL': OS & P1P3 & ANTI_ID_MEDIUM & TRACK_ISOLATION & Q,
    'OS_NONID_NONISOL': OS & P1P3 & ANTI_ID_MEDIUM & TRACK_NONISOLATION & Q,

    'NONISOL': TRACK_NONISOLATION,
}

REGION_SYSTEMATICS = {
    'nOS_NONISOL': 'nOS_ISOL',
    'nOS_ISOL': 'nOS_NONISOL',
    #'nOS': ('nOS_ISOL', 'nOS_NONISOL'),
    'nOS': 'SS',
}
