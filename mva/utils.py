import sys
import os
import errno
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import numpy as np
from rootpy.context import preserve_current_directory
import ROOT


def print_hist(hist):

    print
    if hist.title:
        print hist.title
    for ibin in xrange(len(hist)):
        print "%.5f +/- %.5f" % (hist[ibin], hist.yerrh(ibin))
    print


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


def std(X):

    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)


def rec_to_ndarray(rec, fields=None):

    if fields is None:
        fields = rec.dtype.names
    # Creates a copy and recasts data to a consistent datatype
    return np.vstack([rec[field] for field in fields]).T


def braindump(outdir, indir=None):
    """
    Write out all objects in indir into outdir
    """
    if indir is None:
        indir = ROOT.gDirectory
    with preserve_current_directory():
        outdir.cd()
        for thing in indir.GetList():
            thing.Write()
