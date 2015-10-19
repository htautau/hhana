from rootpy.tree import Cut

__all__ = [ 
    'get_trigger',
]


TRIG_HH_1 = Cut('HLT_tau35_medium1_tracktwo_tau25_medium1_tracktwo_L1TAU20IM_2TAU12IM == 1')
TRIG_HH_2 = Cut('HLT_tau35_loose1_tracktwo_tau25_loose1_tracktwo_L1TAU20IM_2TAU12IM == 1')
TRIG_HH = TRIG_HH_1 #| TRIG_HH_2

TRIG_LH_1 = Cut('HLT_mu26_imedium == 1')
TRIG_LH_2 = Cut('HLT_e28_lhtight_iloose == 1')
TRIG_LH = TRIG_LH_1 | TRIG_LH_2

def get_trigger(channel='hadhad'):
    if channel == 'hadhad':
        return TRIG_HH
    elif channel == 'lephad':
        return TRIG_LH
    else:
        raise RuntimeError('wrong channel name')
