from rootpy.plotting import Hist
from rootpy.interactive import wait
from mva.stats.smooth import smooth, smooth_alt
import ROOT; ROOT.gROOT.SetBatch(False)
import numpy as np


nom = Hist(50, 0, 100, linewidth=2)
nom.fill_array(np.random.uniform(0, 100, 1000))

sys = nom.Clone(linewidth=2, linestyle='dashed')
nom.Smooth(10)

for i, bin in enumerate(sys.bins()):
    bin.value += i / 10.

smooth_sys = smooth(nom, sys, 10,
        linecolor='red',
        linewidth=2,
        linestyle='dashed')
smooth_alt_sys = smooth_alt(nom, sys,
        linecolor='blue',
        linewidth=2,
        linestyle='dashed')

nom.SetMaximum(max(
    nom.GetMaximum(),
    sys.GetMaximum(),
    smooth_sys.GetMaximum(),
    smooth_alt_sys.GetMaximum()) * 1.2)
nom.SetMinimum(-1)

nom.Draw('hist')
sys.Draw('hist same')
smooth_sys.Draw('hist same')
smooth_alt_sys.Draw('hist same')

wait()
