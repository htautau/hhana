# stdlib imports
import sys
import os
import errno
import datetime
from cStringIO import StringIO

# matplotlib imports
from matplotlib.backends.backend_pdf import PdfPages

# ROOT/rootpy
from rootpy.context import preserve_current_directory
import ROOT

# local imports
from . import log; log = log[__name__]


def print_hist(hist):
    out = StringIO()
    print >> out
    print >> out
    if hist.title:
        print >> out, hist.title
    for bin in hist.bins():
        print >> out, "%.5f +/- %.5f" % (bin.value, bin.error)
    log.info(out.getvalue())


def hist_to_dict(hist):
    hist_dict = dict()
    for i, value in enumerate(hist):
        hist_dict[hist.xaxis.GetBinLabel(i + 1)] = value
    return hist_dict


class Tee(object):
    """
    http://stackoverflow.com/questions/616645/
    how-do-i-duplicate-sys-stdout-to-a-log-file-in-python/3423392#3423392
    """
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)


def make_multipage_pdf(figs, name, dir=None):
    now = datetime.datetime.today()
    if dir is not None:
        path = os.path.join(dir, '%s.pdf' % name)
    else:
        path = '%s.pdf' % name
    pdf = PdfPages(path)
    for fig in figs:
        pdf.savefig(fig)
    d = pdf.infodict()
    # set pdf metadata
    d['Title'] = name
    d['Author'] = 'Noel Dawe'
    d['Subject'] = name
    d['Keywords'] = 'higgs tau bdt analysis'
    d['CreationDate'] = now
    d['ModDate'] = now
    pdf.close()


def braindump(outdir, indir=None, func=None):
    """
    Write out all objects in indir into outdir
    """
    llog = log['braindump']
    if indir is None:
        indir = ROOT.gDirectory
    with preserve_current_directory():
        outdir.cd()
        for thing in indir.GetList():
            if func is not None and not func(thing):
                continue
            llog.info("writing {0} in {1}".format(
                thing.GetName(),
                outdir.GetName()))
            thing.Write()


def ravel_hist(hist):
    if hist.GetDimension() != 2:
        return hist
    # convert to 1D hist
    rhist = hist.ravel(name = hist.name + '_ravel')
    if hasattr(hist, 'systematics'):
        # ravel the systematics
        rsyst = {}
        for term, syshist in hist.systematics.items():
            rsyst[term] = syshist.ravel(name=syshist.name + '_ravel')
        rhist.systematics = rsyst
    return rhist


def uniform_hist(hist):
    rhist = hist.uniform_binned(name = hist.name + '_uniform')
    if hasattr(hist, 'systematics'):
        rsyst = {}
        for term, syshist in hist.systematics.items():
            rsyst[term] = syshist.uniform_binned(name=syshist.name + '_uniform')
        rhist.systematics = rsyst
    return rhist


def fold_overflow(hist):
    hist[1] += hist[0]
    hist[-2] += hist[-1]
    if hasattr(hist, 'systematics'):
        for term, syshist in hist.systematics.items():
            syshist[1] += syshist[0]
            syshist[-2] += syshist[-1]


def search_flat_bins(bkg_scores, min_score, max_score, bins):
    scores = []
    weights = []
    for bkg, scores_dict in bkg_scores:
        s, w = scores_dict['NOMINAL']
        scores.append(s)
        weights.append(w)
    scores = np.concatenate(scores)
    weights = np.concatenate(weights)

    selection = (min_score <= scores) & (scores < max_score)
    scores = scores[selection]
    weights = weights[selection]

    sort_idx = np.argsort(scores)
    scores = scores[sort_idx]
    weights = weights[sort_idx]

    total_weight = weights.sum()
    bin_width = total_weight / bins

    # inefficient linear search for now
    weights_cumsum = np.cumsum(weights)
    boundaries = [min_score]
    curr_total = bin_width
    for i, cs in enumerate(weights_cumsum):
        if cs >= curr_total:
            boundaries.append((scores[i] + scores[i+1])/2)
            curr_total += bin_width
        if len(boundaries) == bins:
            break
    boundaries.append(max_score)
    return boundaries
