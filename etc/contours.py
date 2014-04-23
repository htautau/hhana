
import ROOT
from matplotlib import pyplot as plt
from rootpy import asrootpy
from rootpy.plotting import Hist2D, Canvas, set_style
from rootpy.interactive import wait
import numpy as np
from array import array
from mva.plotting import draw_contours

set_style('ATLAS')

a = Hist2D(20, -3, 3, 20, 0, 6)
a.fill_array(np.random.multivariate_normal(
    mean=(0, 3),
    cov=np.arange(4).reshape(2, 2),
    size=(1E6,)))

c = Canvas()
draw_contours(a, linewidths=2, linecolors='red', linestyles='solid',
              labelsizes=18, labelcolors='red', labelformats='%i')
c.Update()
c.SaveAs('plot.png')
wait()
