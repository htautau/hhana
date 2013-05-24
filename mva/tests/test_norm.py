from mva.analysis import Analysis
from mva.categories import Category_Preselection
from nose.tools import assert_almost_equal

def test_norm():

    ss = Analysis(
        year=2012,
        systematics=False,
        use_embedding=True,
        qcd_shape_region='SS_TRK')

    nos = Analysis(
        year=2012,
        systematics=False,
        use_embedding=True,
        qcd_shape_region='nOS')

    ss.normalize(Category_Preselection, 'TRACK')
    nos.normalize(Category_Preselection, 'TRACK')

    qcd_ss = ss.qcd.events(Category_Preselection, 'OS_TRK')
    qcd_nos = nos.qcd.events(Category_Preselection, 'OS_TRK')

    # check that nOS and SS QCD models have the same number of events
    # up to two places after the decimal
    assert_almost_equal(qcd_ss, qcd_nos, places=2)
