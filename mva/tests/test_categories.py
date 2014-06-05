from . import log; log = log[__name__]
from operator import and_

from mva.samples import Data
from mva.categories import CATEGORIES, Category, Category_Preselection
from mva.categories.common import DETA_TAUS

from nose.tools import assert_equal, assert_true


def check_category_orthogonality(year, cat_type):
    data = Data(year)
    cats = CATEGORIES[cat_type]
    cuts = reduce(and_, [c.get_cuts(year) for c in cats])
    log.info(str(cuts))
    assert_equal(data.events(cuts=cuts)[1].value, 0)


def test_category_orthogonality():
    for year in (2011, 2012):
        for cat_type in ('mva_all', 'cuts'):
            yield check_category_orthogonality, year, cat_type


def check_mva_category_sum(year):
    data = Data(year)
    evt_sum = 0.
    for cat in CATEGORIES['mva_all']:
        cur_sum = data.events(cat)[1].value
        assert_true(cur_sum > 0)
        evt_sum += cur_sum
    assert_true(evt_sum > 0)
    presel = data.events(Category_Preselection)[1].value
    presel_deta = data.events(Category_Preselection, cuts=DETA_TAUS)[1].value
    assert_equal(presel_deta, evt_sum)
    assert_true(presel > evt_sum)


def test_mva_category_sum():
    for year in (2011, 2012):
        yield check_mva_category_sum, year
