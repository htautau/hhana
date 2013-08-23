# https://github.com/rootpy/rootpy
from rootpy.plotting import Hist, Legend, Canvas
from rootpy.plotting.style import set_style
from rootpy.interactive import wait
from rootpy.extern.argparse import ArgumentParser
from rootpy import log; log.basic_config_colorized()
import numpy as np

parser = ArgumentParser()
parser.add_argument('--smooth-iterations', type=int, default=1)
parser.add_argument('--transform-scale', type=float, default=5.)
parser.add_argument('--events', type=int, default=1000)
parser.add_argument('--bins', type=int, default=20)
args = parser.parse_args()

set_style('ATLAS')

nominal_scores = np.random.normal(-.3, .2, size=args.events)
up_scores = np.random.normal(-.25, .2, size=args.events)
dn_scores = np.random.normal(-.35, .2, size=args.events)

def transform(x):
    return 2.0 / (1.0 + np.exp(- args.transform_scale * x)) - 1.0

nominal = Hist(args.bins, -1, 1, title='Nominal')
nominal.fill_array(transform(nominal_scores))

up = nominal.Clone(title='Up', linecolor='red', linestyle='dashed', linewidth=2)
up.Reset()
up.fill_array(transform(up_scores))

dn = nominal.Clone(title='Down', linecolor='blue', linestyle='dashed', linewidth=2)
dn.Reset()
dn.fill_array(transform(dn_scores))

# Plot the nominal, up, and down scores

canvas = Canvas()
nominal.SetMaximum(max(dn) * 1.1)
nominal.Draw()
nominal.xaxis.SetTitle('BDT Score')
nominal.yaxis.SetTitle('Events')

up.Draw('same hist')
dn.Draw('same hist')

leg = Legend(3, pad=canvas, leftmargin=.5)
leg.AddEntry(nominal, style='LEP')
leg.AddEntry(up, style='L')
leg.AddEntry(dn, style='L')
leg.Draw()
canvas.SaveAs('canvas_original.png')

# Take the ratio of systematic / nominal

ratio_up = up / nominal
ratio_dn = dn / nominal

ratio_canvas = Canvas()
ratio_canvas.SetLogy()
ratio_up.SetMinimum(0.001)
ratio_up.Draw('hist')
ratio_dn.Draw('same hist')
ratio_up.xaxis.SetTitle('BDT Score')
ratio_up.yaxis.SetTitle('Systematic / Nominal')
ratio_canvas.SaveAs('canvas_ratio.png')

# Now smooth each ratio

ratio_up_smooth = ratio_up.Clone()
ratio_up_smooth.Smooth(args.smooth_iterations)
ratio_dn_smooth = ratio_dn.Clone()
ratio_dn_smooth.Smooth(args.smooth_iterations)

ratio_smooth_canvas = Canvas()
ratio_smooth_canvas.SetLogy()
ratio_up_smooth.SetMinimum(0.001)
ratio_up_smooth.Draw('hist')
ratio_dn_smooth.Draw('same hist')
ratio_up_smooth.xaxis.SetTitle('BDT Score')
ratio_up_smooth.yaxis.SetTitle('Smoothed(Systematic / Nominal)')
ratio_smooth_canvas.SaveAs('canvas_smooth_ratio.png')

# Then scale nominal by the smoothed ratio

up_smooth = ratio_up_smooth * nominal
up_smooth.title = 'Smoothed Up'
dn_smooth = ratio_dn_smooth * nominal
dn_smooth.title = 'Smoothed Down'

canvas_smooth = Canvas()
nominal.SetMaximum(max(dn_smooth) * 1.1)
nominal.Draw()
nominal.xaxis.SetTitle('BDT Score')
nominal.yaxis.SetTitle('Events')

up_smooth.Draw('same hist')
dn_smooth.Draw('same hist')

leg = Legend(3, pad=canvas_smooth, leftmargin=.4)
leg.AddEntry(nominal, style='LEP')
leg.AddEntry(up_smooth, style='L')
leg.AddEntry(dn_smooth, style='L')
leg.Draw()
canvas_smooth.SaveAs('canvas_smooth.png')

wait()
