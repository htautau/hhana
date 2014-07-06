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
from .significance import significance
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


def get_significance_workspace(workspace,
                               observed=False, profile=False,
                               verbose=False, **fit_params):
    if isinstance(profile, basestring):
        if profile == 'hat':
            if not observed:
                fit_params.setdefault('print_level', -1)
            workspace.fit(**fit_params)
            poi = workspace.obj('ModelConfig').GetParametersOfInterest().first()
            profile_mu = poi.getVal()
            profile = True
        else:
            profile_mu = float(profile)
            profile = True
    elif profile is None:
        profile = False
        profile_mu = 1.
    elif profile in (False, True):
        profile_mu = 1.
    else:
        profile_mu = float(profile)
        profile = True
    hist = asrootpy(significance(
        workspace, observed, profile, profile_mu, verbose))
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
