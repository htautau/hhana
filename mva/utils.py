import sys
import os
import errno
from matplotlib.backends.backend_pdf import PdfPages
import datetime


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
