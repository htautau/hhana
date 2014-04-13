from mva.categories import (
    Category_Preselection,
    Category_Cuts_VBF_Preselection,
    Category_Cuts_Boosted_Preselection)
from mva.analysis import Analysis
from mva.variables import VARIABLES, get_label
from mva.plotting import draw_contours
from mva.defaults import TARGET_REGION

from rootpy.plotting import Canvas, Hist2D, set_style
from root_numpy import fill_hist

set_style('ATLAS', shape='rect')

analysis = Analysis(2012)
analysis.normalize(Category_Preselection)
qcd = analysis.qcd
ztt = analysis.ztautau
higgs = analysis.higgs_125
higgs.hist_decor['color'] = 'red'
samples = [qcd, ztt, higgs]
styles = ['solid', 'dashed', 'dotted']

bounds = {
    'mmc1_mass': (0, 200),
    'resonance_pt': (0, 300),
    'dR_tau1_tau2': (0.8, 2.4),
    'dEta_jets': (2.6, 6),
    'mass_jet1_jet2': (0, 1400),
}
scales = {
    'mmc1_mass': 1,
    'resonance_pt': 1E-3,
    'dR_tau1_tau2': 1,
    'dEta_jets': 1,
    'mass_jet1_jet2': 1E-3,
}


def sample_contours(category, x, y, bins=10):
    xbounds, xscale = bounds[x], scales[x]
    ybounds, yscale = bounds[y], scales[y]
    # draw axes
    axes = Hist2D(bins, xbounds[0], xbounds[1],
                  bins, ybounds[0], ybounds[1])
    axes.xaxis.title = get_label(x)
    axes.yaxis.title = get_label(y)
    axes.Draw('AXIS')
    for sample, style in zip(samples, styles):
        array = sample.array(category, TARGET_REGION, fields=[x, y])
        array[:,0] *= xscale
        array[:,1] *= yscale
        hist = axes.Clone()
        fill_hist(hist, array[:,0:2], array[:,-1])
        # normalize
        hist /= hist.integral()
        draw_contours(hist, labelsizes=14,
                      labelcontours=False,
                      linecolors=sample.hist_decor['color'],
                      linewidths=2, linestyles=style,
                      same=True)


canvas = Canvas()
sample_contours(Category_Cuts_VBF_Preselection, 'mmc1_mass', 'resonance_pt')
canvas.SaveAs('contour_vbf_mass_pt.png')

canvas = Canvas()
sample_contours(Category_Cuts_VBF_Preselection, 'mmc1_mass', 'dR_tau1_tau2')
canvas.SaveAs('contour_vbf_mass_dr.png')

canvas = Canvas()
sample_contours(Category_Cuts_VBF_Preselection, 'dR_tau1_tau2', 'resonance_pt')
canvas.SaveAs('contour_vbf_dr_pt.png')

canvas = Canvas()
sample_contours(Category_Cuts_VBF_Preselection, 'dEta_jets', 'mass_jet1_jet2')
canvas.SaveAs('contour_vbf_deta_mjj.png')

