import os
import sys

LIMITS_DIR = os.getenv('HIGGSTAUTAU_LIMITS_DIR')
if not LIMITS_DIR:
    sys.exit('You did not source setup.sh!')
LIMITS_DIR = os.path.join(LIMITS_DIR, 'hadhad')

NTUPLE_PATH = os.path.join(os.getenv('HIGGSTAUTAU_NTUPLE_DIR'), 'prod')
DEFAULT_STUDENT = 'HHProcessor'

BASE_DIR = os.getenv('HIGGSTAUTAU_MVA_DIR')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)

import ROOT
import rootpy
import logging

ROOT.gROOT.SetBatch(True)
rootpy.log.basic_config_colorized()

log = logging.getLogger('mva')
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)

if hasattr(logging, 'captureWarnings'):
    logging.captureWarnings(True)

PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

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

MMC_VERSION = 0
MMC_MASS = 'mmc%d_mass' % MMC_VERSION
MMC_PT = 'mmc%d_resonance_pt' % MMC_VERSION
