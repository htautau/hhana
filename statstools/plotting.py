# python imports
from itertools import cycle
import os
import pickle
import re

# root/rootpy imports
from rootpy import ROOT
from rootpy.plotting import Canvas, Legend, Hist, Graph
from rootpy.plotting.shapes import Line
from rootpy.plotting.utils import draw
from rootpy.memory import keepalive
from rootpy.context import preserve_current_canvas

# local imports
from mva import CACHE_DIR
from mva.samples import Higgs
from . import log; log = log[__name__]

gaussian_cdf_c = ROOT.Math.gaussian_cdf_c

UNBLIND = {
    2012: {
        'vbf': 3,
        'boosted': 1},
    2011: {
        'vbf': 2,
        'boosted': 2}
}


PATTERNS = [
    re.compile('^(?P<type>workspace|channel)(_hh)?_(?P<year>\d+)_(?P<category>[a-z]+)_(?P<mass>\d+)$'),
    re.compile('^(?P<type>workspace|channel)(_hh)?_(?P<category>[a-z]+)_(?P<mass>\d+)_(?P<year>\d+)$')
]


def parse_name(name):
    """
    determine year, category and mass from ws/channel name
    """
    for p in PATTERNS:
        match = re.match(p, name)
        if match:
            break
    if not match:
        raise ValueError(
            "not a valid workspace/channel name: {0}".format(name))
    return (int(match.group('year')) % 1000 + 2000,
            match.group('category'),
            int(match.group('mass')))


def get_data(pickle_file):
    # read NP pull data from a pickle
    with open(pickle_file) as f:
        data = pickle.load(f)
    return data


def print_np(np):
    # strip unneeded text from NP names
    return np.replace('alpha_', '').replace('ATLAS_', '').replace('_', ' ')


def get_rebinned_hist(hist_origin, binning=None):
    if binning is None:
        return hist_origin
    hist_rebin = Hist(binning, name=hist_origin.name+'_rebinned')
    hist_rebin[:] = hist_origin[:]
    return hist_rebin


def get_rebinned_graph(graph_origin, binning=None, unblind=True):
    if binning is None:
        return graph_origin
    log.info(list(graph_origin.x()))
    log.info('Binning: {0}'.format(binning))
    graph_rebin = Graph(len(binning)-1)
    length_filled = len(graph_rebin)
    if (unblind is not True) and isinstance(unblind, int):
        length_filled -= unblind
    log.info('Length: {0} - {1}'.format(len(graph_rebin), length_filled))
    if len(graph_origin) != len(graph_rebin):
        log.info('uniform: {0} bins != rebinned: {1} bins'.format(len(graph_origin), len(graph_rebin)))
        raise RuntimeError('wrong binning')
    else:
        for ip, (y, yerr) in enumerate(zip(graph_origin.y(), graph_origin.yerr())):
            x_rebin_err = 0.5*(binning[ip+1]-binning[ip])
            x_rebin = binning[ip] + x_rebin_err
            graph_rebin.SetPoint(ip, x_rebin, y)
            graph_rebin.SetPointError(ip, x_rebin_err, x_rebin_err, yerr[0], yerr[1])
            if ip>=length_filled:
                graph_rebin.SetPoint(ip, x_rebin, 0)
                graph_rebin.SetPointError(ip, 0, 0, 0, 0)
    return graph_rebin

<<<<<<< HEAD

def get_category(category_name, categories):
    for category in categories:
        if category.name == category_name:
            return category


def get_binning(category, year, fit_var='mmc'):
=======
def get_cat_name(ws_name):
    mass = get_mass(ws_name)
    year = get_year(ws_name)
    if 'channel_' in ws_name:
        cat_name = ws_name.replace('channel_', '')
        words = cat_name.split('_')
        cat_name = '_'.join(words[0:words.index(str(year%1000))])
    elif 'hh_{0}'.format(year%1000) in ws_name:
        cat_name = ws_name.replace('hh_{0}'.format(year%1000), '')
        words = cat_name.split('_')
        cat_name = '_'.join(words[1:words.index(str(mass))])
    else:
        cat_name = ws_name
    return cat_name

def get_category(ws_name, categories):
    cat_name = get_cat_name(ws_name)
    for cat in categories:
        if cat.name==cat_name:
            return cat
        
def get_year(ws_name):
    if '12' in ws_name.split('_'):
        return 2012
    elif '11' in ws_name.split('_'):
        return 2011
    else:
        return None

def get_blinding(name):
    year = get_year(name)
    if 'vbf' in name:
        return UNBLIND[year]['vbf']
    elif 'boost' in name:
        return UNBLIND[year]['boosted']
    else:
        return True

def get_mass(ws_name):
    masses = Higgs.MASSES
    words = ws_name.split('_')
    for mass in masses:
        if str(mass) in words:
            return mass

def get_binning(name, categories, fit_var='mmc'):
    binning = []
    year = get_year(name)
    cat = get_category(name, categories)
    mass = get_mass(name)
    log.info('Year: {0}; Mass: {1}; Category: {2}'.format(year, mass, cat.name))
    if fit_var == 'mmc':
        binning = category.limitbins
        if isinstance(binning, (tuple, list)):
            binning[-1] = 250
            return binning
        binning[year][-1] = 250
        return binning[year]
    if fit_var == 'bdt':
        binning_cache = os.path.join(
            CACHE_DIR, 'binning/binning_{0}_{1}_{2}.pickle'.format(
                category.name, 125, year % 1000))
        if os.path.exists(binning_cache):
            with open(binning_cache) as f:
                binning = pickle.load(f)
            return binning
    # use original binning in WS
    return None


def get_blinding(category, year, fit_var='mmc'):
    if fit_var == 'mmc':
        return (100, 150)
    return UNBLIND[category.name][year]


def get_uncertainty_graph(hnom, curve_uncert):
    """
    Convert an histogram and a RooCurve
    into a TGraphAsymmError

    Parameters
    ----------
    hnom: TH1F, TH1D, ...
        The histogram of nominal values
    curve_uncert: RooCurve
        The uncertainty band around the nominal value

    TODO: Improve the handling of the underflow and overflow bins
    """
    graph = Graph(hnom.GetNbinsX())
    for ibin in xrange(1, hnom.GetNbinsX() + 1):
        uncerts = []
        for ip in xrange(3, curve_uncert.GetN() - 3):
            x, y = ROOT.Double(0.), ROOT.Double(0.)
            curve_uncert.GetPoint(ip, x, y)
            if hnom.GetBinLowEdge(ibin) <= x < hnom.GetBinLowEdge(ibin + 1):
                uncerts.append(y)
        log.info('{0}, bin {1}: {2}'.format(hnom.name, ibin, uncerts))
        low, high = min(uncerts), max(uncerts)
        bin_center = 0.5 * (hnom.GetBinLowEdge(ibin + 1) +
                            hnom.GetBinLowEdge(ibin))
        e_x_low = bin_center - hnom.GetBinLowEdge(ibin)
        e_x_high = hnom.GetBinLowEdge(ibin + 1) - bin_center
        bin_content = hnom.GetBinContent(ibin)
        e_y_low = hnom.GetBinContent(ibin) - low
        e_y_high = high - hnom.GetBinContent(ibin)
        graph.SetPoint(ibin - 1, bin_center, bin_content)
        graph.SetPointError(ibin - 1, e_x_low, e_x_high, e_y_low, e_y_high)
    return graph


def pvalue_plot(poi, pvalues, pad=None,
                xtitle='X', ytitle='P_{0}',
                linestyle=None,
                linecolor=None,
                yrange=None,
                verbose=False):
    """
    Draw a pvalue plot

    Parameters
    ----------
    poi : list
        List of POI values tested
    pvalues : list
        List of p-values or list of lists of p-values to overlay
        multiple p-value curves
    pad : Canvas or Pad, optional (default=None)
        Pad to draw onto. Create new pad if None.
    xtitle : str, optional (default='X')
        The x-axis label (POI name)
    ytitle : str, optional (default='P_{0}')
        The y-axis label
    linestyle : str or list, optional (default=None)
        Line style for the p-value graph or a list of linestyles for
        multiple p-value graphs.
    linecolor : str or list, optional (default=None)
        Line color for the p-value graph or a list of linestyles for
        multiple p-value graphs.

    Returns
    -------
    pad : Canvas
        The pad.
    graphs : list of Graph
        The p-value graphs

    """
    if not pvalues:
        raise ValueError("pvalues is empty")
    if not poi:
        raise ValueError("poi is empty")
    # determine if pvalues is list or list of lists
    if not isinstance(pvalues[0], (list, tuple)):
        pvalues = [pvalues]
    if linecolor is not None:
        if not isinstance(linecolor, list):
            linecolor = [linecolor]
        linecolor = cycle(linecolor)
    if linestyle is not None:
        if not isinstance(linestyle, list):
            linestyle = [linestyle]
        linestyle = cycle(linestyle)

    with preserve_current_canvas():
        if pad is None:
            pad = Canvas()
        pad.cd()
        pad.SetLogy()

        # create the axis
        min_poi, max_poi = min(poi), max(poi)
        haxis = Hist(1000, min_poi, max_poi)
        xaxis = haxis.xaxis
        yaxis = haxis.yaxis
        xaxis.SetRangeUser(min_poi, max_poi)
        haxis.Draw('AXIS')

        min_pvalue = float('inf')
        graphs = []
        for ipv, pv in enumerate(pvalues):
            graph = Graph(len(poi), linestyle='dashed',
                          drawstyle='L', linewidth=2)
            for idx, (point, pvalue) in enumerate(zip(poi, pv)):
                graph.SetPoint(idx, point, pvalue)
            if linestyle is not None:
                graph.linestyle = linestyle.next()
            if linecolor is not None:
                graph.linecolor = linecolor.next()
            graphs.append(graph)
            curr_min_pvalue = min(pv)
            if curr_min_pvalue < min_pvalue:
                min_pvalue = curr_min_pvalue

        if verbose:
            for graph in graphs:
                log.info(['{0:1.1f}'.format(xval) for xval in list(graph.x())])
                log.info(['{0:0.3f}'.format(yval) for yval in list(graph.y())])

        # automatically handles axis limits
        axes, bounds = draw(graphs, pad=pad, same=True, logy=True,
             xtitle=xtitle, ytitle=ytitle,
             xaxis=xaxis, yaxis=yaxis, ypadding=(0.2, 0.1),
             logy_crop_value=1E-300)

        if yrange is not None:
            xaxis, yaxis = axes
            yaxis.SetLimits(*yrange)
            yaxis.SetRangeUser(*yrange)
            min_pvalue = yrange[0]

        # draw sigma levels up to minimum of pvalues
        line = Line()
        line.SetLineStyle(2)
        line.SetLineColor(2)
        latex = ROOT.TLatex()
        latex.SetNDC(False)
        latex.SetTextSize(20)
        latex.SetTextColor(2)
        sigma = 0
        while True:
            pvalue = gaussian_cdf_c(sigma)
            if pvalue < min_pvalue:
                break
            keepalive(pad, latex.DrawLatex(max_poi, pvalue, " {0}#sigma".format(sigma)))
            keepalive(pad, line.DrawLine(min_poi, pvalue, max_poi, pvalue))
            sigma += 1

        pad.RedrawAxis()
        pad.Update()
    return pad, graphs


if __name__ == '__main__':
    from rootpy.plotting import Canvas, Legend, get_style
    from rootpy.plotting.style.atlas.labels import ATLAS_label

    mass_points = [100,105,120,125,130,135,140,145,150]
    pvalues = [
        [0.5, 0.25, 0.15, 0.05, 0.03, 0.01, 0.03, 0.05, 0.15, 0.25, 0.5],
        [0.4, 0.3, 0.17, 0.02, 0.01, 0.008, 0.08, 0.06, 0.14, 0.2, 0.2],
    ]
    names = ['A', 'B']
    style = get_style('ATLAS', shape='rect')
    # allow space for sigma labels on right
    style.SetPadRightMargin(0.05)
    with style:
        c = Canvas()
        _, graphs = pvalue_plot(
            mass_points, pvalues, pad=c, xtitle='m_{H} [GeV]',
            linestyle=['dashed', 'solid'])
        for name, graph in zip(names, graphs):
            graph.title = name
            graph.legendstyle = 'L'
        leg = Legend(graphs, leftmargin=0.4, topmargin=0.2)
        leg.Draw()
        ATLAS_label(0.57, 0.88, text="Internal 2012", sqrts=8, pad=c, sep=0.09)
        c.SaveAs('pvalue_plot.png')
