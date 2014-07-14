from rootpy import asrootpy
from .extern import AsymptoticsCLs


def asymptotic_CLs(workspace, observed=False, verbose=False):
    calculator = AsymptoticsCLs(workspace, verbose)
    hist = asrootpy(calculator.run('ModelConfig', 'obsData', 'asimovData'))
    hist.SetName('%s_limit' % workspace.GetName())
    return hist
