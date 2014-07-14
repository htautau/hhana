# default minimizer options
from rootpy.utils.silence import silence_sout_serr
with silence_sout_serr():
    from rootpy.stats import mute_roostats; mute_roostats()

import ROOT
ROOT.Math.MinimizerOptions.SetDefaultStrategy(1)
ROOT.Math.MinimizerOptions.SetDefaultMinimizer('Minuit2')

import os
import logging

log = logging.getLogger('statstools')
if not os.environ.get('DEBUG', False):
    log.setLevel(logging.INFO)
