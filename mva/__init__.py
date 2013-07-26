import os
import sys

LIMITS_DIR = os.getenv('HIGGSTAUTAU_LIMITS_DIR')
if not LIMITS_DIR:
    sys.exit('You did not source setup.sh!')

import ROOT
import rootpy
import logging

LIMITS_DIR = os.path.join(LIMITS_DIR, 'hadhad')

NTUPLE_PATH = os.path.join(os.getenv('HIGGSTAUTAU_NTUPLE_DIR'), 'prod')
DEFAULT_STUDENT = 'hhskim'

if not os.getenv('MVA_NO_BATCH', False):
    ROOT.gROOT.SetBatch(True)
rootpy.log.basic_config_colorized()

log = logging.getLogger('mva')
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)

if hasattr(logging, 'captureWarnings'):
    logging.captureWarnings(True)

BASE_DIR = os.getenv('HIGGSTAUTAU_MVA_DIR')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

if not os.path.exists(CACHE_DIR):
    log.info("creating directory %s" % CACHE_DIR)
    os.mkdir(CACHE_DIR)

PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

if not os.path.exists(PLOTS_DIR):
    log.info("creating directory %s" % PLOTS_DIR)
    os.mkdir(PLOTS_DIR)

from .utils import mkdir_p

def plots_dir(script):

    script = os.path.basename(script)
    script = os.path.splitext(script)[0]
    dir = os.path.join(PLOTS_DIR, script)
    if not os.path.exists(dir):
        mkdir_p(dir)
    return dir

import numpy as np
# for reproducibilty
# especially for test/train set selection
np.random.seed(1987) # my birth year ;)

MMC_VERSION = 1
MMC_MASS = 'mmc%d_mass' % MMC_VERSION
MMC_PT = 'mmc%d_resonance_pt' % MMC_VERSION

from rootpy.plotting.style import get_style, set_style

style = get_style('ATLAS')
#style.SetFrameLineWidth(2)
style.SetLineWidth(2)
style.SetTitleYOffset(1.8)
style.SetTickLength(0.04, 'X')
style.SetTickLength(0.02, 'Y')
style.SetHistTopMargin(0.)
style.SetHatchesLineWidth(2)
set_style(style)

#ROOT.TGaxis.SetMaxDigits(3)

from rootpy.fit import mute_roostats; mute_roostats()
