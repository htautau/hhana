import rootpy
import os
import logging

log = logging.getLogger('statstools')
if not os.environ.get("DEBUG", False):
    log.setLevel(logging.INFO)

from rootpy import asrootpy
from rootpy.plotting import Hist
from rootpy.stats import histfactory

from .asymptotics import AsymptoticsCLs
from .significance import runSig


def get_limit(channels,
          unblind=False,
          lumi=1.,
          lumi_rel_error=0.,
          POI='SigXsecOverSM'):
    workspace, _ = histfactory.make_workspace('higgs', channels,
        lumi=lumi,
        lumi_rel_error=lumi_rel_error,
        POI=POI,
        #silence=True
        )
    return get_limit_workspace(workspace, unblind=unblind)


def get_limit_workspace(workspace, unblind=False, verbose=False):
    calculator = AsymptoticsCLs(workspace, verbose)
    hist = asrootpy(calculator.run('ModelConfig', 'obsData', 'asimovData'))
    hist.SetName('%s_limit' % workspace.GetName())
    return hist


def get_significance_workspace(workspace, blind=True, verbose=False):
    hist = asrootpy(runSig(workspace, blind,verbose))
    hist.SetName('%s_significance' % workspace.GetName())
    return hist
