from statstools import get_significance_workspace
from rootpy.plotting import Hist, Canvas
from rootpy.plotting import root2matplotlib as rplt
from rootpy.stats.histfactory import (
    Data, Sample, Channel, make_measurement, make_workspace)
from root_numpy import fill_hist
import numpy as np
import matplotlib.pyplot as plt
from math import log

from mva import CONST_PARAMS
from mva.categories import Category_VBF
from mva.analysis import Analysis

from statstools.fixups import fix_measurement

"""
def transform(x, c=1.):
    if c == 0:
        return x
    return 2.0 / (1.0 + np.exp(- c * x)) - 1.0

b = transform(np.random.normal(-0.2, .2, 5000), 7)
s = transform(np.random.normal(0.2, .2, 100), 7)

def optimize():
    # initial binning defines one bin from min to max value
    binning = [min_x, max_x]
    binning_min = binning[:]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for nbins in xrange(8):
        left, right = binning[0], binning[1]
        edges = np.linspace(left, right, 100, endpoint=False)[1:]
        sigs = [get_sig(s, b, binning, x) for x in edges]
        ax.plot(edges, sigs)
        best_edge = edges[np.argmax(sigs)]
        binning.insert(1, best_edge)

        def func(x):
            return -get_sig(s, b, binning_min, x)

        edge, _ = search(func, binning_min[0], binning_min[1])
        binning_min.insert(1, edge)

    fig.savefig('test.png')

    canvas = Canvas()
    signal = Hist(binning, color='red', drawstyle='hist')
    background = Hist(binning, color='blue', drawstyle='hist')
    fill_hist(signal, *s)
    fill_hist(background, *b)
    signal.Draw()
    background.Draw('same')
    canvas.SaveAs('test2.png')

    print binning
    print binning_min

    print get_sig(s, b, binning)
    print get_sig(s, b, binning_min)


def get_best_edge(s, b, edges, pos=0, iter=50):
    left, right = edges[pos:pos+2]
    probe_edges = np.linspace(left, right, iter, endpoint=False)[1:]
    sigs = [get_sig(s, b, edges, x, pos+1) for x in probe_edges]
    best_sig = np.max(sigs[10:])
    best_edge = probe_edges[np.argmax(sigs[10:]) + 10]
    return probe_edges, sigs, best_edge, best_sig


def get_best_bin(s, b, left, right, iter=50):
    probe_edges = np.linspace(left, right, iter, endpoint=False)[1:]
    sigs = [get_sig(s, b, [x, right]) for x in probe_edges]
    best_sig = np.max(sigs[10:])
    best_edge = probe_edges[np.argmax(sigs[10:]) + 10]
    return probe_edges, sigs, best_edge, best_sig

"""

def search(f, a, b, tol=1.0e-9, **kwargs):
    """
    Golden section method for determining x that minimizes
    the user-supplied scalar function f(x).
    The minimum must be bracketed in (a,b).
    """
    nIter = int(-2.078087*log(tol/abs(b-a)))
    R = 0.618033989
    C = 1.0 - R
    # First telescoping
    x1 = R*a + C*b; x2 = C*a + R*b
    f1 = f(x1, **kwargs); f2 = f(x2, **kwargs)
    # Main loop
    for i in range(nIter):
        if f1 > f2:
            a = x1
            x1 = x2; f1 = f2
            x2 = C*a + R*b; f2 = f(x2, **kwargs)
        else:
            b = x2
            x2 = x1; f2 = f1
            x1 = R*a + C*b; f1 = f(x1, **kwargs)
    if f1 < f2:
        return x1, f1
    else:
        return x2, f2


def get_workspace(scores, binning, fix=False):
    hist_template = Hist(binning)
    background = []
    for sample, scores_dict in scores.bkg_scores:
        background.append(sample.get_histfactory_sample(
            hist_template, None, category, 'OS', scores=scores_dict))
    signal = []
    for sample, scores_dict in scores.all_sig_scores[125]:
        signal.append(sample.get_histfactory_sample(
            hist_template, None, category, 'OS', scores=scores_dict))
    data_hist = sum([b.hist for b in background])
    data_hist.name = 'Data'
    data = Data('Data', data_hist)
    channel = Channel(category.name, signal + background, data)
    measurement = make_measurement('MVA', channel,
        POI='SigXsecOverSM',
        const_params=CONST_PARAMS)
    if fix:
        fix_measurement(measurement)
    return make_workspace(measurement, silence=False)


def get_sig(scores, binning, edge=None, pos=1):
    if edge is not None:
        binning = binning[:]
        binning.insert(pos, edge)
    ws = get_workspace(scores, binning)
    hist = get_significance_workspace(ws)
    sig = hist[2].value
    # handle nan
    return 0 if sig != sig else sig


def optimize_func(edge, scores, binning):
    return - get_sig(scores, binning, edge, 1)


systematics = False
transform = False

category = Category_VBF
analysis = Analysis(2012, transform=transform, systematics=systematics)
analysis.normalize(category)
clf = analysis.get_clf(category, load=True)
scores = analysis.get_scores(
    clf, category, 'OS', mode='workspace', mass_points=[125], systematics=True)

# nominal scores for convenience
b = np.concatenate([scores_dict['NOMINAL'][0] for _, scores_dict in scores.bkg_scores])
bw = np.concatenate([scores_dict['NOMINAL'][1] for _, scores_dict in scores.bkg_scores])
s = np.concatenate([scores_dict['NOMINAL'][0] for _, scores_dict in scores.all_sig_scores[125]])
sw = np.concatenate([scores_dict['NOMINAL'][1] for _, scores_dict in scores.all_sig_scores[125]])
min_score = min(np.min(s), np.min(b)) - 1E-8
max_score = max(np.max(s), np.max(b)) + 1E-8
s = (s, sw)
b = (b, bw)

# poor man's constant width binning
nfixed_bins = range(1, 41)
fixed_sigs = []
for bins in nfixed_bins:
    fixed_sigs.append(get_sig(
        scores, np.linspace(min_score, max_score, bins + 1)))
max_fixed_sig = np.max(fixed_sigs)
max_fixed_nbins = nfixed_bins[np.argmax(fixed_sigs)]

# demonstrate smart binning
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Significance')
ax1.set_xlabel('BDT Score')
ax2 = ax1.twiny()
ax2.set_xlabel('Number of Fixed-width Bins')
ax3 = ax1.twinx()
ax3.set_ylabel('Events')
ax3.set_yscale('log')
#ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# plot the distributions
b_hist = Hist(20, min_score, max_score, color='blue',
              linewidth=3, linestyle='dashed')
s_hist = b_hist.Clone(color='red')
fill_hist(b_hist, *b)
fill_hist(s_hist, *s)
rplt.hist(b_hist, axes=ax3, label='Background')
rplt.hist(s_hist, axes=ax3, label='Signal')

from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

# show significance vs middle bin edge location
binning = [min_score, max_score]
for nbins in xrange(5):
    #edges, sigs, best_edge, best_sig = get_best_edge(s, b, binning)
    #ax1.plot(edges, sigs, color='black', linestyle=next(linecycler))
    best_edge, best_sig = search(optimize_func, binning[0], binning[1],
        scores=scores, binning=binning)
    binning.insert(1, best_edge)
    ax1.plot((best_edge, best_edge), (0, best_sig),
        color='black', linestyle='-', linewidth=2)

# show significance vs number of equal width bins
ax2.plot(nfixed_bins, fixed_sigs, label='Fixed-width Bins', color='green', linestyle='-')

#handles1, labels1 = ax1.get_legend_handles_labels()
#handles2, labels2 = ax2.get_legend_handles_labels()
#handles3, labels3 = ax3.get_legend_handles_labels()
#ax2.legend(handles1+handles2+handles3, labels1+labels2+labels3)
plt.tight_layout()
fig.savefig('test3.png')

"""
if False:
    _, _, best_edge, best_sig = get_best_edge(s, b, [min_x, max_x])

    right = max_x
    all_edges = [min_x, max_x]
    for nbins in xrange(10):
        _, _, edge, sig = get_best_bin(s, b, min_x, right)
        right = edge
        all_edges.insert(1, edge)
        sig_total = get_sig(s, b, all_edges)
        print sig, sig_total, max_fixed_nbins, max_fixed_sig
        print best_edge, best_sig
        print all_edges
        raw_input()


if False:
    edges, sigs, best_edge, best_sig = get_best_edge(s, b, [min_x, max_x])

    for nbins in range(2, 301):
        _edges = list(np.linspace(min_x, best_edge, nbins)) + [max_x]
        _sig = get_sig(s, b, _edges)

        #edges = [min_x] + list(np.linspace(best_edge, max_x, nbins))
        #sig = get_sig(s, b, edges)

        edges = np.linspace(min_x, max_x, nbins)
        sig = get_sig(s, b, edges)
        print _sig
        print _edges

        print sig
        print edges
        raw_input()
"""
