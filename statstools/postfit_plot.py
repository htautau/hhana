# python imports
import os
# root/rootpy imports
from rootpy import ROOT
from rootpy.plotting import Hist, Graph
# local imports
from mva import CACHE_DIR
from mva.samples import Higgs
from . import log; log = log[__name__]

#----------------------
def get_rebinned_hist(hist_origin, binning):
    hist_rebin = Hist(binning, name=hist_origin.name+'_rebinned') 
    hist_rebin[:] = hist_origin[:]
    return hist_rebin
    
#----------------------
def get_rebinned_graph(graph_origin, binning):
    log.info(list(graph_origin.x()))
    log.info(binning)
    graph_rebin = Graph(len(binning)-1)
    if len(graph_origin) != len(graph_rebin):
        log.info('{0} != {1}'.format(len(graph_origin), len(graph_rebin)))
        raise RuntimeError('wrong binning')
    else:
        for ip, (y, yerr) in enumerate(zip(graph_origin.y(), graph_origin.yerr())):
            x_rebin_err = 0.5*(binning[ip+1]-binning[ip])
            x_rebin = binning[ip] + x_rebin_err
            graph_rebin.SetPoint(ip, x_rebin, y)
            graph_rebin.SetPointError(ip, x_rebin_err, x_rebin_err, yerr[0], yerr[1])
    return graph_rebin                     

#-----------------------
def get_category(ws_cat_name, categories):
    for cat in categories:
        if cat.name in ws_cat_name:
            return cat
#-----------------------
def get_year(ws_name):
    if '_12' and not '_11' in ws_name:
        return 2012
    else:
        return 2011

#-----------------------
def get_mass(ws_name):
    masses = Higgs.MASSES
    for mass in masses:
        if '_{0}_'.format(mass) in ws_name:
            return mass

#----------------------------
def get_binning(name, categories, fit_var='mmc_mass'):
    binning = []
    cat = get_category(name, categories)
    year = get_year(name)
    mass = get_mass(name)
    log.info(year)
    log.info(mass)
    log.info(cat)
    if fit_var=='mmc_mass':
        binning = cat.limitbins
        if isinstance(binning, (tuple, list)):
            binning[-1] = 250
            return binning
        else:
            binning[year][-1] = 250
            return binning[year]
    else:
        with open(os.path.join(CACHE_DIR, 'binning/binning_{0}_{1}_{2}.pickle'.format(
            cat.name, mass, year % 1000))) as f:
            binning = pickle.load(f)
            binning[0] += 1E5
            binning[-1] -= 1E5
            return binning
# -------------------------------------
def UncertGraph(hnom, curve_uncert):
    """
    Convert an histogram and a RooCurve
    into a TGraphAsymmError

    Parameters
    ----------
    hnom: TH1F, TH1D,...
        The histogram of nominal values
    curve_uncert: RooCurve
        The uncertainty band around the nominal value
    curve_uncert: RooCurve
    TODO: Improve the handling of the underflow and overflow bins
    """

    graph = Graph(hnom.GetNbinsX())
    # ---------------------------------------------
    for ibin in xrange(1, hnom.GetNbinsX()+1):
        uncerts = []
        for ip in xrange(3, curve_uncert.GetN()-3):
            x, y = ROOT.Double(0.), ROOT.Double(0.)
            curve_uncert.GetPoint(ip, x, y)
            if int(x)==int(hnom.GetBinLowEdge(ibin)):
                uncerts.append(y)
        uncerts.sort()
        log.info('{0}: {1}'.format(hnom.name, uncerts))
        if len(uncerts) !=2:
            for val in uncerts:
                if val in uncerts:
                    uncerts.remove(val) 
        if len(uncerts)!=2:
            raise RuntimeError('Need exactly two error values and got {0}'.format(uncerts))

        bin_center = 0.5*(hnom.GetBinLowEdge(ibin+1)+hnom.GetBinLowEdge(ibin))
        e_x_low = bin_center-hnom.GetBinLowEdge(ibin)
        e_x_high = hnom.GetBinLowEdge(ibin+1) - bin_center
        bin_content = hnom.GetBinContent(ibin)
        e_y_low = hnom.GetBinContent(ibin)-uncerts[0]
        e_y_high = uncerts[1]-hnom.GetBinContent(ibin) 
        graph.SetPoint( ibin-1, bin_center, bin_content)
        graph.SetPointError(ibin-1, e_x_low, e_x_high, e_y_low, e_y_high)
    # ---------------------------------------------
    return graph
