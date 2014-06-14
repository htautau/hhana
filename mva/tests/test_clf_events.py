from mva.analysis import Analysis
from mva.samples import Higgs
from mva.categories import Category_VBF, Category_Boosted, Category_Preselection
from mva.defaults import TARGET_REGION
from rootpy.plotting import Hist
from root_numpy import fill_hist

from numpy.testing import assert_almost_equal


def check_sample_category(analysis, sample, category):
    clf = analysis.get_clf(category, mass=125, load=True)
    scores, weights = sample.scores(
        clf, category, TARGET_REGION,
        systematics=False)['NOMINAL']
    hist = Hist(20, scores.min() - 1E-5, scores.max() + 1E-5)
    fill_hist(hist, scores, weights)
    assert_almost_equal(sample.events(category, TARGET_REGION)[1].value,
                        hist.integral(), 3)


def test_clf_events():
    for year in 2011, 2012:
        higgs = Higgs(year, mass=125)
        analysis = Analysis(year)
        analysis.normalize(Category_Preselection)
        qcd = analysis.qcd
        for sample in (qcd, higgs):
            for category in (Category_VBF, Category_Boosted):
                yield check_sample_category, analysis, sample, category
