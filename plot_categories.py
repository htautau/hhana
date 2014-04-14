from mva.categories import (
    Category_Preselection,
    Category_VBF, Category_Boosted,
    Category_Cuts_VBF_Preselection,
    Category_Cuts_Boosted_Preselection)
from mva.analysis import Analysis
from mva.variables import VARIABLES, get_label
from mva.plotting import draw_contours
from mva.defaults import TARGET_REGION
from mva import save_canvas

from rootpy.plotting import Canvas, Hist, Hist2D, Profile2D, Legend, set_style
from root_numpy import fill_hist, fill_profile
import numpy as np

set_style('ATLAS', shape='rect')

analysis = Analysis(2012)
analysis.normalize(Category_Preselection)
qcd = analysis.qcd
ztt = analysis.ztautau
higgs = analysis.higgs_125
higgs.hist_decor['color'] = 'red'
background = [qcd, ztt]
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


def bdt_contours(mva_category, category, x, y, bins=10):
    clf = analysis.get_clf(mva_category, load=True, mass=125)
    xbounds, xscale = bounds[x], scales[x]
    ybounds, yscale = bounds[y], scales[y]
    # draw axes
    profile = Profile2D(bins, xbounds[0], xbounds[1],
                        bins, ybounds[0], ybounds[1])
    profile.xaxis.title = get_label(x)
    profile.yaxis.title = get_label(y)
    for sample in samples:
        array = sample.array(category, TARGET_REGION, fields=[x, y])
        scores_dict = sample.scores(clf, category, TARGET_REGION)
        scores, weights = scores_dict['NOMINAL']
        scores = np.reshape(scores, (-1, 1))
        array[:,0] *= xscale
        array[:,1] *= yscale
        #assert((weights == array[:,-1]).all())
        array = np.c_[array[:,0], array[:,1], scores]
        # ignore negative weights
        array = array[weights > 0]
        weights = weights[weights > 0]
        fill_profile(profile, array)#, weights)
    draw_contours(profile, labelsizes=14,
                  labelcontours=True,
                  linecolors='black',
                  linewidths=2, linestyles='dashed')

def legend(canvas, right=False):
    objs = [Hist(1, 0, 1, legendstyle='L',
                 linewidth=2, linestyle=style,
                 linecolor=s.hist_decor['color'],
                 title=s.label)
            for s, style in zip(samples, styles)]
    legend = Legend(objs, pad=canvas, textsize=20,
                    leftmargin=0.5 if right else 0.04,
                    rightmargin=0.04 if right else 0.5,
                    margin=0.3)
    legend.Draw()


formats = ('.eps', '.png')

# VBF
canvas = Canvas()
sample_contours(Category_Cuts_VBF_Preselection, 'mmc1_mass', 'resonance_pt')
legend(canvas)
save_canvas(canvas, 'plots/contours', 'contour_vbf_mass_pt', formats)

canvas = Canvas()
bdt_contours(Category_VBF, Category_Cuts_VBF_Preselection, 'mmc1_mass', 'resonance_pt')
save_canvas(canvas, 'plots/contours', 'contour_vbf_mass_pt_bdt', formats)

canvas = Canvas()
sample_contours(Category_Cuts_VBF_Preselection, 'mmc1_mass', 'dR_tau1_tau2')
legend(canvas)
save_canvas(canvas, 'plots/contours', 'contour_vbf_mass_dr', formats)

canvas = Canvas()
bdt_contours(Category_VBF, Category_Cuts_VBF_Preselection, 'mmc1_mass', 'dR_tau1_tau2')
save_canvas(canvas, 'plots/contours', 'contour_vbf_mass_dr_bdt', formats)

canvas = Canvas()
sample_contours(Category_Cuts_VBF_Preselection, 'dR_tau1_tau2', 'resonance_pt')
legend(canvas, right=True)
save_canvas(canvas, 'plots/contours', 'contour_vbf_dr_pt', formats)

canvas = Canvas()
bdt_contours(Category_VBF, Category_Cuts_VBF_Preselection, 'dR_tau1_tau2', 'resonance_pt')
save_canvas(canvas, 'plots/contours', 'contour_vbf_dr_pt_bdt', formats)

canvas = Canvas()
sample_contours(Category_Cuts_VBF_Preselection, 'dEta_jets', 'mass_jet1_jet2')
legend(canvas)
save_canvas(canvas, 'plots/contours', 'contour_vbf_deta_mjj', formats)

canvas = Canvas()
bdt_contours(Category_VBF, Category_Cuts_VBF_Preselection, 'dEta_jets', 'mass_jet1_jet2')
save_canvas(canvas, 'plots/contours', 'contour_vbf_deta_mjj_bdt', formats)

# Boosted
canvas = Canvas()
sample_contours(Category_Cuts_Boosted_Preselection, 'mmc1_mass', 'resonance_pt')
legend(canvas)
save_canvas(canvas, 'plots/contours', 'contour_boosted_mass_pt', formats)

canvas = Canvas()
bdt_contours(Category_Boosted, Category_Cuts_Boosted_Preselection, 'mmc1_mass', 'resonance_pt')
save_canvas(canvas, 'plots/contours', 'contour_boosted_mass_pt_bdt', formats)

canvas = Canvas()
sample_contours(Category_Cuts_Boosted_Preselection, 'mmc1_mass', 'dR_tau1_tau2')
legend(canvas)
save_canvas(canvas, 'plots/contours', 'contour_boosted_mass_dr', formats)

canvas = Canvas()
bdt_contours(Category_Boosted, Category_Cuts_Boosted_Preselection, 'mmc1_mass', 'dR_tau1_tau2')
save_canvas(canvas, 'plots/contours', 'contour_boosted_mass_dr_bdt', formats)

canvas = Canvas()
sample_contours(Category_Cuts_Boosted_Preselection, 'dR_tau1_tau2', 'resonance_pt')
legend(canvas)
save_canvas(canvas, 'plots/contours', 'contour_boosted_dr_pt', formats)

canvas = Canvas()
bdt_contours(Category_Boosted, Category_Cuts_Boosted_Preselection, 'dR_tau1_tau2', 'resonance_pt')
save_canvas(canvas, 'plots/contours', 'contour_boosted_dr_pt_bdt', formats)