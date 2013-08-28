from mva.stats.hypotests import optimize_binning
from mva.tests.utils import get_background_signal
from rootpy.plotting import Canvas

b, s = get_background_signal(scale=5, sig_events=10000, bkg_events=10000)
before = Canvas()
b.SetMaximum(max(max(b), max(s)) * 1.1)
b.Draw()
s.Draw('same')
before.SaveAs('binning_before.png')

s, b, _ = optimize_binning(s, b, starting_point='fine')
after = Canvas()
b.SetMaximum(max(max(b), max(s)) * 1.1)
b.Draw()
s.Draw('same')
after.SaveAs('binning_after.png')
