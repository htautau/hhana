import numpy as np
from random import gauss, random
from rootpy.plotting import Hist
from mva.stats.utils import uniform_binning, zero_negs, kylefix
from nose.tools import assert_equal, assert_almost_equal


def test_uniform_binning():

    a = Hist([0, 10, 100, 10000, 100000])
    for v in [5, 20, 200, 20000]:
        a.Fill(v, gauss(10, 5))
    b = uniform_binning(a)
    assert_equal(list(a), list(b))
    assert_equal(list(a.yerrh()), list(b.yerrh()))

def test_zero_negs():

    a = Hist(100, 0, 1)

    for i in xrange(1000):
        a.Fill(random(), gauss(.3, 5))
    assert_equal((np.array(list(a)) < 0).any(), True)
    b = zero_negs(a)
    assert_equal((np.array(list(b)) < 0).any(), False)

def test_kylefix():

    a = Hist(10, 0, 1)
    for w in xrange(1, 11):
        a.Fill(.5, w)
    b = kylefix(a)
    assert_equal(b[0], 5.5)
