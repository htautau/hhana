
from style import set_hsg4_style
from rootpy.plotting import Hist
from templates import SimplePlot, RatioPlot

import ROOT
ROOT.gROOT.SetBatch()

set_hsg4_style()

hist = Hist(10, 0, 1)
hist.FillRandom('gaus')

plot = SimplePlot(xtitle='X [GeV]', ytitle='Events')
plot.draw('main', hist)
plot.SaveAs('simple.png')

plot = RatioPlot(xtitle='X [GeV]', ytitle='Events', ratio_title='Data / Model')
plot.draw('main', hist)
plot.SaveAs('ratio.png')
