from rootpy.plotting import Hist
from mva.samples import Embedded_Ztautau, MC_Ztautau
from mva.categories import Category_Preselection
from nose.tools import assert_equals, assert_almost_equal

samples = [
        Embedded_Ztautau(2012),
        MC_Ztautau(2012)
]

for a in samples:
    events = a.events(Category_Preselection, 'OS', raw=True)
    events_weighted = a.events(Category_Preselection, 'OS')
    rec = a.merged_records(Category_Preselection, 'OS')
    hist = Hist(1, -100, 100)
    a.draw_into(hist, 'tau1_pt>-100', Category_Preselection, 'OS')
    assert_equals(events, rec.shape[0])
    assert_almost_equal(events_weighted, rec['weight'].sum(), places=1)
    assert_almost_equal(events_weighted, hist.Integral(), places=1)
