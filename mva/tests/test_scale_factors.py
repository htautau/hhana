from rootpy.tree import Cut
from mva.samples import *
from mva.categories import Category_VBF
from mva.defaults import TARGET_REGION
from nose.tools import assert_equal, assert_greater

tau1_matched = Cut('tau1_matched')
tau2_matched = Cut('tau2_matched')
both_taus_matched = tau1_matched & tau2_matched


def check_matching(sample):
    print sample.label
    total_events = sample.events(Category_VBF, TARGET_REGION)[1].value
    print "total events: ", total_events
    print "tau1 matched: ", sample.events(Category_VBF, TARGET_REGION, cuts=tau1_matched)[1].value / total_events
    print "tau2 matched: ", sample.events(Category_VBF, TARGET_REGION, cuts=tau2_matched)[1].value / total_events
    print "both matched: ", sample.events(Category_VBF, TARGET_REGION, cuts=both_taus_matched)[1].value / total_events


def check_fakerate(sample):
    not_matched = sample.events(Category_VBF, TARGET_REGION, cuts=-tau1_matched)[1].value
    matched = sample.events(Category_VBF, TARGET_REGION, cuts=tau1_matched)[1].value
    sf_applied = sample.events(Category_VBF, TARGET_REGION, cuts='tau1_fakerate_sf < 1')[1].value
    sf_not_applied = sample.events(Category_VBF, TARGET_REGION, cuts='tau1_fakerate_sf == 1')[1].value
    #assert_greater(not_matched, 0)
    assert_greater(matched, 0)
    assert_equal(not_matched, sf_applied)
    assert_equal(matched, sf_not_applied)


def test_matching_and_fakerates():
    ztautau = MC_Ztautau(year=2012)
    #yield check_matching, ztautau
    yield check_fakerate, ztautau
    others = Others(2012)
    yield check_fakerate, others
    signals = []
    for mass in Higgs.MASSES:
        for mode in Higgs.MODES:
            signal = Higgs(year=2012, mass=mass, mode=mode)
            #yield check_matching, signal
            yield check_fakerate, signal
