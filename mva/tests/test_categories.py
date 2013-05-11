from . import log; log = log[__name__]
from operator import and_

from mva.samples import Data
from mva.categories import CATEGORIES, Category

from nose.tools import assert_equal


def test_category_orthogonality():

    data = Data(year=2012)
    cats = CATEGORIES['harmonize']

    class CombinedCat(Category):
        cuts = reduce(and_, [c.cuts for c in cats])
        year_cuts = cats[0].year_cuts
    log.info(str(CombinedCat.cuts))
    assert_equal(data.events(CombinedCat, 'OS_TRK'), 0)
