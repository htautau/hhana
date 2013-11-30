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


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST:
            pass
        else: raise


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


def ravel(hist):
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
