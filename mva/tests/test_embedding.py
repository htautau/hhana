from mva.samples import Embedded_Ztautau, MC_Ztautau
from mva.categories import Category_Preselection, Category_VBF, Category_Boosted
from nose.tools import assert_equals, assert_almost_equal

embed = Embedded_Ztautau(2012, systematics=False)
mc = MC_Ztautau(2012, systematics=False)

categories = [
    Category_Preselection,
    Category_Boosted,
    Category_VBF
]

def test_charge():

    print
    for category in categories:
        print category.name
        print "embed: OS / SS = ", embed.events(category, 'OS_TRK') / embed.events(category, 'SS_TRK')
        print "mc: OS / SS = ", mc.events(category, 'OS_TRK') / mc.events(category, 'SS_TRK')

