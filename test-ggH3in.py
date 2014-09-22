from mva.samples import Higgs
from mva.analysis import Analysis
from mva.categories import Category_VBF
from rootpy.plotting import Hist, Canvas
from rootpy.plotting.utils import draw

ana = Analysis(2012)
clf = ana.get_clf(Category_VBF, load=True)
h = Higgs(2012, mode='gg', systematics=True)
sample = h.get_histfactory_sample(
    Hist(clf.binning(2012), color='red',
         fillstyle='hollow', drawstyle='hist',
         linewidth=2),
    clf, Category_VBF, 'OS_ISOL',
    systematics=True, mva=True)
hsys = sample.GetHistoSys('QCDscale_ggH3in')

nom = sample.hist.Clone(shallow=True, drawstyle='hist', color='red', fillstyle='hollow', linewidth=2)
up = hsys.high.Clone(shallow=True, drawstyle='hist', color='red', linestyle='dashed', fillstyle='hollow', linewidth=2)
dn = hsys.low.Clone(shallow=True, drawstyle='hist', color='red', linestyle='dotted', fillstyle='hollow', linewidth=2)

canvas = Canvas()
draw([nom, up, dn])
canvas.SaveAs('test_ggH3in.png')
