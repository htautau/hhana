from rootpy.plotting import Hist
from mva.analysis import Analysis
from mva.categories import Category_VBF
from nose.tools import assert_equal, assert_almost_equal
try:
    from nose.tools import assert_multi_line_equal
except ImportError:
    assert_multi_line_equal = assert_equal
else:
    assert_multi_line_equal.im_class.maxDiff = None


analysis = Analysis(year=2012, systematics=True, use_embedding=True)
analysis.normalize(Category_VBF)


def test_draw():

    for sample in analysis.backgrounds:
        print sample.name

        hist = Hist(30, 0, 250)
        hist_array = hist.Clone()
        field_hist = {'mmc1_mass': hist_array}

        sample.draw_into(hist, 'mmc1_mass', Category_VBF, 'OS_TRK')
        sample.draw_array(field_hist, Category_VBF, 'OS_TRK')

        assert_almost_equal(hist.Integral(), hist_array.Integral(), places=3)

        assert_equal(sorted(hist.systematics.keys()),
                     sorted(hist_array.systematics.keys()))

        for term, sys_hist in hist.systematics.items():
            print term
            sys_hist_array = hist_array.systematics[term]
            assert_almost_equal(sys_hist.Integral(), sys_hist_array.Integral())
