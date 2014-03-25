from statstools import get_significance_workspace
from rootpy.plotting import Hist, Canvas
from rootpy.plotting import root2matplotlib as rplt
from rootpy.stats.histfactory import (
    Data, Sample, Channel, make_workspace)
from root_numpy import fill_hist
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from math import log

def transform(x, c=1.):
    if c == 0:
        return x
    return 2.0 / (1.0 + np.exp(- c * x)) - 1.0

b = transform(np.random.normal(-0.2, .2, 5000), 7)
s = transform(np.random.normal(0.2, .2, 100), 7)

min_x = min(np.min(s), np.min(b)) - 1E-8
max_x = max(np.max(s), np.max(b)) + 1E-8


from mva.categories import Category_VBF
from mva.analysis import Analysis

analysis = Analysis(2012)
analysis.normalize(Category_VBF)
clf = analysis.get_clf(Category_VBF)




def search(f, a, b, tol=1.0e-9):
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
    f1 = f(x1); f2 = f(x2)
    # Main loop
    for i in range(nIter):
        if f1 > f2:
            a = x1
            x1 = x2; f1 = f2
            x2 = C*a + R*b; f2 = f(x2)

        else:
            b = x2
            x2 = x1; f2 = f1
            x1 = R*a + C*b; f1 = f(x1)
    if f1 < f2:
        return x1,f1
    else:
        return x2,f2


def workspace(s, b, binning, fix=True):
    signal_hist = Hist(binning)
    fill_hist(signal_hist, s)
    signal = Sample('Signal', signal_hist.uniform_binned())
    signal.AddNormFactor('SigXsecOverSM', 0., 0., 200., False)
    background_hist = Hist(binning)
    fill_hist(background_hist, b)
    if fix:
        for bin in background_hist.bins(overflow=False):
            if bin.value <= 0:
                bin.value = 1E-5
                bin.error = 1E8
    background = Sample('Background', background_hist.uniform_binned())
    background.ActivateStatError()
    background.AddOverallSys('Systematic', 1.1, 0.9)
    data_hist = background_hist.Clone()
    data = Data('Data', data_hist.uniform_binned())
    channel = Channel('Analysis', [signal, background], data)
    return make_workspace(
        'workspace', [channel],
        POI='SigXsecOverSM',
        const_params=['Lumi'],
        silence=False)[0]


def get_sig(s, b, binning, edge=None, pos=1):
    if edge is not None:
        binning = binning[:]
        binning.insert(pos, edge)
    ws = workspace(s, b, binning)
    hist = get_significance_workspace(ws)
    sig = hist[2].value
    # nan
    return 0 if sig != sig else sig


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
    fill_hist(signal, s)
    fill_hist(background, b)
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


nfixed_bins = range(1, 31)
fixed_sigs = []
for bins in nfixed_bins:
    fixed_sigs.append(get_sig(s, b, np.linspace(min_x, max_x, bins + 1)))
max_fixed_sig = np.max(fixed_sigs)
max_fixed_nbins = nfixed_bins[np.argmax(fixed_sigs)]


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

if True:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Significance')
    ax1.set_xlabel('Toy BDT Score')
    ax2 = ax1.twiny()
    ax2.set_xlabel('Number of Fixed-width Bins')
    ax3 = ax1.twinx()
    ax3.set_ylabel('Events')
    #ax3.set_yscale('log')
    #ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    # plot the distributions
    b_hist = Hist(20, min_x, max_x, color='blue', linewidth=3, linestyle='dashed')
    s_hist = b_hist.Clone(color='red')
    fill_hist(b_hist, b)
    fill_hist(s_hist, s)
    rplt.hist(b_hist, axes=ax3, label='Background')
    rplt.hist(s_hist, axes=ax3, label='Signal')

    from itertools import cycle
    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)

    # show significance vs middle bin edge location
    binning = [min_x, max_x]
    for nbins in xrange(10):
        edges, sigs, best_edge, best_sig = get_best_edge(s, b, binning)
        ax1.plot(edges, sigs, color='black', linestyle=next(linecycler))
        binning.insert(1, best_edge)
        ax1.plot((best_edge, best_edge), (0, best_sig), color='black', linestyle='-')

    # show significance vs number of equal width bins
    ax2.plot(nfixed_bins, fixed_sigs, label='Fixed-width Bins', color='green', linestyle='-')

    #handles1, labels1 = ax1.get_legend_handles_labels()
    #handles2, labels2 = ax2.get_legend_handles_labels()
    #handles3, labels3 = ax3.get_legend_handles_labels()
    #ax2.legend(handles1+handles2+handles3, labels1+labels2+labels3)
    plt.tight_layout()
    fig.savefig('test3.png')
