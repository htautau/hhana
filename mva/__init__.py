import os
import sys

BASE_DIR = os.getenv('HIGGSTAUTAU_MVA_DIR')
if not BASE_DIR:
    sys.exit('You did not source setup.sh!')

CACHE_DIR = os.path.join(BASE_DIR, 'cache')
if not os.path.exists(CACHE_DIR):
    log.info("creating directory %s" % CACHE_DIR)
    os.mkdir(CACHE_DIR)

ETC_DIR = os.path.join(BASE_DIR, 'etc')
DAT_DIR = os.path.join(BASE_DIR, 'dat')
BDT_DIR = os.path.join(BASE_DIR, 'bdts')
NTUPLE_PATH = os.path.join(os.getenv('HIGGSTAUTAU_NTUPLE_DIR'), 'prod_v29')

import ROOT
import rootpy
import logging

# Speed things up a bit
ROOT.SetSignalPolicy(ROOT.kSignalFast)

DEFAULT_STUDENT = 'hhskim'

if not os.getenv('MVA_NO_BATCH', False):
    ROOT.gROOT.SetBatch(True)

log = logging.getLogger('mva')
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)

if hasattr(logging, 'captureWarnings'):
    logging.captureWarnings(True)

log['/ROOT.TH1D.Chi2TestX'].setLevel(log.WARNING)

from rootpy.utils.path import mkdir_p

def plots_dir(script):
    script = os.path.basename(script)
    script = os.path.splitext(script)[0]
    dir = os.path.join(PLOTS_DIR, script)
    mkdir_p(dir)
    return dir

import numpy as np
# for reproducibilty
# especially for test/train set selection
np.random.seed(1987)

MMC_VERSION = 1
MMC_MASS = 'mmc%d_mass' % MMC_VERSION
MMC_PT = 'mmc%d_resonance_pt' % MMC_VERSION

from rootpy.plotting.style import get_style, set_style

style = get_style('ATLAS', shape='square')
#style.SetFrameLineWidth(2)
#style.SetLineWidth(2)
#style.SetTitleYOffset(1.8)
#style.SetTickLength(0.04, 'X')
#style.SetTickLength(0.02, 'Y')

# custom HSG4 modifications
# style.SetPadTopMargin(0.06)
style.SetPadLeftMargin(0.16)
style.SetTitleYOffset(1.6)
style.SetHistTopMargin(0.)
style.SetHatchesLineWidth(1)
style.SetHatchesSpacing(1)
set_style(style)

ROOT.TGaxis.SetMaxDigits(4)

from rootpy.utils.silence import silence_sout_serr
with silence_sout_serr():
    from rootpy.stats import mute_roostats; mute_roostats()

import yellowhiggs
log.info("using yellowhiggs {0}".format(yellowhiggs.__version__))

CONST_PARAMS = [
    'Lumi',
    'mu_XS8_ggH',
    'mu_XS7_ggH',
    'mu_XS8_VBF',
    'mu_XS7_VBF',
    'mu_XS8_WH',
    'mu_XS7_WH',
    'mu_XS8_ZH',
    'mu_XS7_ZH',
    'mu_BR_tautau',
]

POI = 'SigXsecOverSM'

# pip install --user GitPython
from git import Repo
REPO = Repo(BASE_DIR)
REPO_BRANCH = REPO.active_branch
PLOTS_DIR = os.path.join(BASE_DIR, 'plots', 'variables')


def plot_dir(name):
    return os.path.join(BASE_DIR, 'plots', name)


def save_canvas(canvas, directory, name, formats=None):
    # save images in directories corresponding to current git branch
    filepath = os.path.join(directory, REPO_BRANCH, name)
    path = os.path.dirname(filepath)
    if not os.path.exists(path):
        mkdir_p(path)
    if formats is not None:
        for fmt in formats:
            canvas.SaveAs(filepath + fmt)
    else:
        canvas.SaveAs(filepath)
