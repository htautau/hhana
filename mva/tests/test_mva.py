from rootpy.tree import Cut
from mva.samples import *

tau1_matched = Cut('tau1_matched')
tau2_matched = Cut('tau2_matched')
both_taus_matched = tau1_matched & tau2_matched


def matching(sample):

    print sample.label
    total_events = sample.events()
    print "total events: ", total_events
    print "tau1 matched: ", sample.events(tau1_matched) / total_events
    print "tau2 matched: ", sample.events(tau2_matched) / total_events
    print "both matched: ", sample.events(both_taus_matched) / total_events


def fakerate(sample):

    assert sample.events(-tau1_matched) == sample.events('tau1_fakerate_scale_factor < 1')
    assert sample.events(tau1_matched) == sample.events('tau1_fakerate_scale_factor == 1')


def test_mva():

    ztautau = MC_Ztautau(year=2012, systematics=False)
    matching(ztautau)
    fakerate(ztautau)

    signals = []
    for mass in Higgs.MASS_POINTS:
        for mode in Higgs.MODES:
            signal = Higgs(year=2012, mass=mass, mode=mode,
                    systematics=False)
            matching(signal)
            fakerate(signal)

