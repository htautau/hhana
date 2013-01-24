import os

LIMITS_DIR = os.getenv('HIGGSTAUTAU_LIMITS_DIR')
if not LIMITS_DIR:
    sys.exit('You did not source setup.sh!')
LIMITS_DIR = os.path.join(LIMITS_DIR, 'hadhad')

import ROOT
import rootpy
import logging

ROOT.SetBatch(True)
rootpy.log.basic_config_colorized()

log = logging.getLogger('mva')
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)

logging.captureWarnings(True)

PLOTS_DIR = './plots'

from .utils import mkdir_p

def plots_dir(script):

    script = os.path.basename(script)
    script = os.path.splitext(script)[0]
    dir = os.path.join(PLOTS_DIR, script)
    if not os.path.exists(dir):
        mkdir_p(dir)
    return dir

import numpy
# for reproducibilty
# especially for test/train set selection
np.random.seed(1987) # my birth year ;)
