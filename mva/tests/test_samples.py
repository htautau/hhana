from rootpy.plotting import Hist
from mva.samples import Embedded_Ztautau, MC_Ztautau, Data
from mva.categories import Category_Preselection, Category_VBF
from nose.tools import assert_equals, assert_almost_equal

samples = [
    Embedded_Ztautau(2012, systematics=False),
    MC_Ztautau(2012, systematics=False),
    Data(2012)
]

categories = [
    Category_Preselection,
    Category_VBF
]

def test_shape():

    for category in categories:
        for sample in samples:
            events = sample.events(category, 'OS', raw=True)
            events_weighted = sample.events(category, 'OS')
            rec = sample.merged_records(category, 'OS')
            hist = Hist(1, -100, 100)
            sample.draw_into(hist, 'tau1_pt>-100', category, 'OS')
            assert_equals(events, rec.shape[0])
            assert_almost_equal(events_weighted, rec['weight'].sum(), places=1)
            assert_almost_equal(events_weighted, hist.Integral(), places=1)

            left, right = sample.partitioned_records(category, 'OS')
            print left.shape, right.shape, rec.shape
            assert_equals(left.shape[0] + right.shape[0], rec.shape[0])
