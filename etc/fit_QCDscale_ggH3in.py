from rootpy.plotting import Canvas
from rootpy.io import root_open
import ROOT
ROOT.gROOT.SetBatch(True)

func_name = 'pol3'
rebin = 2

with root_open('ShapeUncertainty_HadHad.root', 'read') as input:
    with root_open('QCDscale_ggH3in.root', 'recreate') as output:
        nom = input.h_nominal
        up = input.h_up
        dn = input.h_down
        nom.rebin(rebin)
        up_ratio = up.rebin(rebin) / nom
        dn_ratio = dn.rebin(rebin) / nom

        with Canvas(name='up_ratio_{0:d}'.format(rebin)) as c:
            up_ratio.Fit(func_name, 'WLF')
            func = up_ratio.GetFunction(func_name)
            func.SetName('up_fit')
            func.Write()
            up_ratio.Draw()
            c.save_as('up_ratio_{0:d}.png'.format(rebin))
        with Canvas(name='dn_ratio_{0:d}'.format(rebin)) as c:
            dn_ratio.Fit(func_name, 'WLF')
            func = dn_ratio.GetFunction(func_name)
            func.SetName('dn_fit')
            func.Write()
            dn_ratio.Draw()
            c.save_as('dn_ratio_{0:d}.png'.format(rebin))
