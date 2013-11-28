from mva.samples import Higgs
from mva.categories import Category_VBF
from rootpy.plotting import Hist

fields = {'tau1_pt': Hist(100, 20000, 160000)}
s = Higgs(2012, mass=125)
s.draw_array(fields, Category_VBF, 'nOS')
print list(fields['tau1_pt'])
