# default minimizer options
import ROOT
ROOT.Math.MinimizerOptions.SetDefaultStrategy(1)
ROOT.Math.MinimizerOptions.SetDefaultMinimizer('Minuit2')

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
from .fitresult import Prefit_RooFitResult, Partial_RooFitResult


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


def get_significance_workspace(workspace, blind=True,
                               mu_profile_value=1, verbose=False):
    if mu_profile_value == 'hat':
        workspace.fit()
        poi = workspace.obj('ModelConfig').GetParametersOfInterest().first()
        mu_profile_value = poi.getVal()
    elif isinstance(mu_profile_value, basestring):
        mu_profile_value = float(mu_profile_value)
    hist = asrootpy(runSig(workspace, blind, mu_profile_value, verbose))
    hist.SetName('%s_significance' % workspace.GetName())
    return hist


def get_bestfit_nll_workspace(workspace, return_nll=False):
    if return_nll:
        roo_min, nll_func = asrootpy(workspace).fit(return_nll=return_nll)
        fitres = roo_min.save()
        return fitres.minNll(), nll_func
    else:
        roo_min = asrootpy(workspace).fit(return_nll=return_nll)
        fitres = roo_min.save()
        return fitres.minNll()
