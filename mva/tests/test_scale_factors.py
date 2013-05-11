from rootpy.tree import Cut
from mva.samples import *
from mva.categories import Category_VBF
from nose.tools import assert_equals

tau1_matched = Cut('tau1_matched')
tau2_matched = Cut('tau2_matched')
both_taus_matched = tau1_matched & tau2_matched


def matching(sample):

    print sample.label
    total_events = sample.events(Category_VBF, 'OS_TRK')
    print "total events: ", total_events
    print "tau1 matched: ", sample.events(Category_VBF, 'OS_TRK', cuts=tau1_matched) / total_events
    print "tau2 matched: ", sample.events(Category_VBF, 'OS_TRK', cuts=tau2_matched) / total_events
    print "both matched: ", sample.events(Category_VBF, 'OS_TRK', cuts=both_taus_matched) / total_events


def fakerate(sample):

    assert_equals(sample.events(Category_VBF, 'OS_TRK', cuts=-tau1_matched),
                  sample.events(Category_VBF, 'OS_TRK', cuts='tau1_fakerate_scale_factor < 1'))
    assert_equals(sample.events(Category_VBF, 'OS_TRK', cuts=tau1_matched),
                  sample.events(Category_VBF, 'OS_TRK', cuts='tau1_fakerate_scale_factor == 1'))


def test_fakerates():

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

